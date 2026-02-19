import json
import logging
import os
import re
import subprocess

import numpy as np

log = logging.getLogger(__name__)

DEPRESSION_KEYWORDS = [
    "triste",
    "tristeza",
    "deprimido",
    "deprimida",
    "deprimente",
    "sozinho",
    "sozinha",
    "solidão",
    "vazio",
    "vazia",
    "desesperado",
    "desesperada",
    "sem esperança",
    "desespero",
    "cansado",
    "cansada",
    "exausto",
    "exausta",
    "esgotado",
    "esgotada",
    "não consigo",
    "não aguento",
    "não dá mais",
    "sem sentido",
    "sem propósito",
    "inútil",
    "fracasso",
    "culpa",
    "culpado",
    "culpada",
    "ninguém entende",
    "ninguém se importa",
    "me afastar",
    "isolar",
    "isolamento",
    "não durmo",
    "insônia",
    "sem apetite",
    "sem energia",
    "desistir",
    "acabar com tudo",
    "sumir",
    "angústia",
    "ansiedade",
    "medo",
    "pavor",
    "choro",
    "chorar",
    "não consigo cuidar",
    "não sinto amor",
    "mãe ruim",
    "não quero o bebê",
    "não consigo amamentar",
    "pós-parto",
    "puerpério",
    "baby blues",
    "medo do parto",
    "medo de perder",
    "aborto",
    "sangramento",
    "gestação de risco",
    "prematuridade",
    "me bateu",
    "me agrediu",
    "apanhei",
    "ameaça",
    "ameaçou",
    "tenho medo dele",
    "não me deixa sair",
    "controla",
    "me humilha",
    "violência",
    "abuso",
    "fadiga",
    "hormônio",
    "menopausa",
    "tpm",
    "ciclo irregular",
]

NEGATIVE_PATTERNS = [
    r"\bnão\s+\w+",
    r"\bnunca\b",
    r"\bnada\b",
    r"\bsempre\s+(triste|mal|cansad[oa]|sozinh[oa])",
]

HESITATION_MARKERS = ["é...", "hum", "hmm", "tipo", "assim", "né", "ah", "ahn"]


class AudioAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.audio_path = video_path.rsplit(".", 1)[0] + "_audio.wav"
        self.results = {
            "transcricao": "",
            "keywords_encontradas": [],
            "score_depressao": 0,
            "indicadores": [],
            "hesitacoes": 0,
            "features_voz": {},
        }

    def _extract_audio(self) -> bool:
        if os.path.exists(self.audio_path):
            log.info("Áudio já extraído: %s", self.audio_path)
            return True
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    self.video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    self.audio_path,
                ],
                check=True,
                capture_output=True,
            )
            log.info("Áudio extraído: %s", self.audio_path)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.error("Falha ao extrair áudio: %s", e)
            return False

    def _transcribe(self) -> str:
        try:
            import whisper
        except ImportError:
            log.error("openai-whisper não instalado. Execute: pip install openai-whisper")
            return ""

        import ssl

        import certifi

        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

        log.info("Transcrevendo com Whisper (modelo base)...")
        model = whisper.load_model("base")
        result = model.transcribe(self.audio_path, language="pt")
        text = result.get("text", "")
        self.results["transcricao"] = text
        log.info("Transcrição concluída (%d caracteres)", len(text))
        return text

    def _analyze_text(self, text: str):
        if not text:
            return
        lower = text.lower()
        score = 0
        found = []

        for kw in DEPRESSION_KEYWORDS:
            if kw in lower:
                found.append(kw)
                score += 2

        for pat in NEGATIVE_PATTERNS:
            matches = re.findall(pat, lower)
            if matches:
                self.results["indicadores"].append(f"Padrão negativo: {pat}")
                score += len(matches)

        neg_words = ["não", "nunca", "nada", "nenhum", "nem"]
        neg_count = sum(lower.count(w) for w in neg_words)
        if neg_count > 5:
            self.results["indicadores"].append(f"Alto uso de negações ({neg_count}x)")
            score += int(neg_count * 0.5)

        first_person = ["eu ", " me ", " meu ", " minha ", " mim "]
        fp_count = sum(lower.count(w) for w in first_person)
        if fp_count > 10:
            self.results["indicadores"].append("Foco excessivo em si (possível ruminação)")
            score += 2

        hesitations = sum(lower.count(h) for h in HESITATION_MARKERS)
        if hesitations > 5:
            self.results["indicadores"].append(f"Hesitação frequente ({hesitations}x)")
            score += min(hesitations, 5)
        self.results["hesitacoes"] = hesitations

        self.results["keywords_encontradas"] = found
        self.results["score_depressao"] = score

    def _analyze_voice(self):
        try:
            import librosa
        except ImportError:
            log.warning("librosa não instalado — análise vocal desabilitada")
            return

        y, sr = librosa.load(self.audio_path, sr=16000)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[pitches > 0]
        pitch_mean = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0

        energy = float(np.sum(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        intervals = librosa.effects.split(y, top_db=30)
        gaps = []
        for i in range(1, len(intervals)):
            gap = (intervals[i][0] - intervals[i - 1][1]) / sr
            if gap > 1.0:
                gaps.append(gap)

        self.results["features_voz"] = {
            "pitch_medio": round(pitch_mean, 1),
            "energia": round(energy, 2),
            "zero_crossing_rate": round(zcr, 4),
            "pausas_longas": len(gaps),
        }

        if pitch_mean > 0 and pitch_mean < 120:
            self.results["indicadores"].append("Tom de voz baixo (baixa energia/tristeza)")
            self.results["score_depressao"] += 1
        if energy < 100:
            self.results["indicadores"].append("Baixa energia vocal")
            self.results["score_depressao"] += 1
        if len(gaps) > 5:
            self.results["indicadores"].append(f"Muitas pausas longas ({len(gaps)}x)")
            self.results["score_depressao"] += 2

    def analyze(self, transcription: str = None) -> dict:
        if transcription:
            self.results["transcricao"] = transcription
        else:
            if self._extract_audio():
                transcription = self._transcribe()

        if transcription:
            self._analyze_text(transcription)
            self._analyze_voice()
        else:
            log.warning("Sem transcrição disponível")

        return self.results

    def generate_report(self, output_path: str = "relatorio_audio.json") -> dict:
        r = self.results
        report = {
            "arquivo": self.video_path,
            "transcricao": r["transcricao"][:500] + ("..." if len(r["transcricao"]) > 500 else ""),
            "analise_fala": {
                "score_depressao": r["score_depressao"],
                "nivel": self._nivel(r["score_depressao"]),
                "keywords": r["keywords_encontradas"][:15],
                "indicadores": r["indicadores"],
                "hesitacoes": r["hesitacoes"],
                "features_voz": r["features_voz"],
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        txt = output_path.replace(".json", ".txt")
        self._write_txt(report, txt)
        log.info("Relatório áudio: %s", output_path)
        return report

    @staticmethod
    def _nivel(score):
        if score < 5:
            return "Baixo"
        return "Moderado" if score < 15 else "Alto"

    @staticmethod
    def _write_txt(report, path):
        sep = "=" * 72
        a = report["analise_fala"]
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{sep}\nRELATÓRIO DE ANÁLISE DE ÁUDIO\n{sep}\n\n")
            f.write(f"Arquivo: {report['arquivo']}\n\n")
            f.write(f"--- TRANSCRIÇÃO (trecho) ---\n{report['transcricao']}\n\n")
            f.write(f"--- ANÁLISE ---\nScore: {a['score_depressao']}  Nível: {a['nivel']}\n")
            f.write(f"Hesitações: {a['hesitacoes']}\n\n")
            if a["keywords"]:
                f.write("Palavras-chave:\n")
                for kw in a["keywords"]:
                    f.write(f"  • {kw}\n")
            if a["indicadores"]:
                f.write("\nIndicadores:\n")
                for ind in a["indicadores"]:
                    f.write(f"  • {ind}\n")
            if a["features_voz"]:
                f.write("\nFeatures vocais:\n")
                for k, v in a["features_voz"].items():
                    f.write(f"  • {k}: {v}\n")
            f.write(f"\n{sep}\n")
