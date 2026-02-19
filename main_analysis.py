import json
import logging
import os
import sys
from datetime import datetime

from audio_analysis import AudioAnalyzer
from video_analysis import VideoAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

WEIGHT_VIDEO = 0.4
WEIGHT_AUDIO = 0.6


def classify_risk(score: float) -> str:
    if score < 3:
        return "BAIXO"
    if score < 8:
        return "MODERADO"
    if score < 15:
        return "ALTO"
    return "MUITO ALTO"


def recommendation(score: float, video_report: dict, audio_report: dict | None) -> str:
    lines = []

    if score < 3:
        lines.append("Sem sinais significativos detectados.")
    elif score < 8:
        lines.append("ATENÇÃO: Alguns indicadores presentes.")
        lines.append("  → Converse com pessoas de confiança.")
        lines.append("  → Considere apoio psicológico.")
    elif score < 15:
        lines.append("ALERTA: Múltiplos indicadores detectados.")
        lines.append("  → Procure um profissional de saúde mental.")
        lines.append("  → CVV: 188 (24h, gratuito)")
    else:
        lines.append("URGÊNCIA: Sinais graves detectados.")
        lines.append("  → LIGUE AGORA: CVV 188 ou SAMU 192")
        lines.append("  → Informe familiares/amigos.")

    if video_report["depressao"]["indicadores"]:
        lines.append("\nIndicadores visuais:")
        for i in video_report["depressao"]["indicadores"][:3]:
            lines.append(f"  • {i}")

    if audio_report and audio_report["analise_fala"]["keywords"]:
        lines.append("\nIndicadores na fala:")
        kws = audio_report["analise_fala"]["keywords"][:5]
        lines.append(f"  • Palavras-chave: {', '.join(kws)}")

    return "\n".join(lines)


class IntegratedAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_analyzer = VideoAnalyzer(video_path)
        self.audio_analyzer = AudioAnalyzer(video_path)

    def run(self) -> dict:
        log.info("Etapa 1/2: Análise visual")
        self.video_analyzer.analyze(sample_rate=30)
        video_report = self.video_analyzer.generate_report("relatorio_video.json")

        log.info("Etapa 2/2: Análise de áudio")
        audio_results = self.audio_analyzer.analyze()
        audio_report = None
        if audio_results["transcricao"]:
            audio_report = self.audio_analyzer.generate_report("relatorio_audio.json")

        return self._fuse(video_report, audio_report)

    def _fuse(self, video_report: dict, audio_report: dict | None) -> dict:
        score_v = video_report["depressao"]["score"]
        score_a = audio_report["analise_fala"]["score_depressao"] if audio_report else 0
        score_total = score_v * WEIGHT_VIDEO + score_a * WEIGHT_AUDIO

        integrated = {
            "arquivo": self.video_path,
            "timestamp": datetime.now().isoformat(),
            "video": video_report,
            "audio": audio_report or {"disponivel": False},
            "fusao_multimodal": {
                "depressao": {
                    "score_total": round(score_total, 2),
                    "score_visual": score_v,
                    "score_audio": score_a,
                    "risco": classify_risk(score_total),
                    "recomendacao": recommendation(score_total, video_report, audio_report),
                },
                "violencia_domestica": {
                    "hematomas": video_report["hematomas"]["total"],
                    "score_risco": video_report["hematomas"]["score_risco"],
                    "risco": video_report["hematomas"]["nivel_risco"],
                },
                "saude": {
                    "marcas": video_report["marcas"]["total"],
                    "tipos": video_report["marcas"]["tipos"],
                },
            },
        }

        self._save(integrated)
        return integrated

    @staticmethod
    def _save(data: dict):
        with open("relatorio_integrado.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        fusion = data["fusao_multimodal"]
        sep = "=" * 72
        with open("relatorio_integrado.txt", "w", encoding="utf-8") as f:
            f.write(f"{sep}\nRELATÓRIO INTEGRADO — ANÁLISE MULTIMODAL\n")
            f.write(f"Saúde da Mulher: Depressão, Violência, Problemas de Saúde\n{sep}\n\n")
            f.write(f"Arquivo: {data['arquivo']}\nData: {data['timestamp']}\n\n")

            d = fusion["depressao"]
            f.write("--- DEPRESSÃO (fusão multimodal) ---\n")
            f.write(f"Score total: {d['score_total']}  Risco: {d['risco']}\n")
            f.write(f"  Visual: {d['score_visual']}  Áudio: {d['score_audio']}\n\n")
            f.write(f"Recomendação:\n{d['recomendacao']}\n\n")

            v = fusion["violencia_domestica"]
            f.write("--- VIOLÊNCIA DOMÉSTICA ---\n")
            f.write(f"Hematomas: {v['hematomas']}  Risco: {v['risco']}\n")
            if v["hematomas"] > 0:
                f.write("Central de Atendimento à Mulher: 180\n")

            s = fusion["saude"]
            f.write(f"\n--- SAÚDE ---\nMarcas: {s['marcas']}\n")

            f.write(f"\n{sep}\nRECURSOS DE APOIO\n{sep}\n")
            f.write("CVV (saúde mental): 188\n")
            f.write("Central da Mulher (violência): 180\n")
            f.write("SAMU: 192\n")
            f.write(f"\n{sep}\n")
            f.write("Triagem automatizada — NÃO substitui avaliação profissional.\n")
            f.write(f"{sep}\n")

        log.info("Relatório integrado salvo")


def main():
    video_path = "data/YTDown.com_YouTube_Depoimento-Tratamento-Haoma-Depressao_Media_6wUi9bANsuE_002_720p.mp4"

    if not os.path.exists(video_path):
        log.error("Vídeo não encontrado: %s", video_path)
        sys.exit(1)

    analyzer = IntegratedAnalyzer(video_path)
    results = analyzer.run()

    fusion = results["fusao_multimodal"]
    log.info("─── RESULTADO ───")
    log.info("Depressão: %s (score %.2f)", fusion["depressao"]["risco"], fusion["depressao"]["score_total"])
    log.info(
        "Violência: %s (%d hematomas)",
        fusion["violencia_domestica"]["risco"],
        fusion["violencia_domestica"]["hematomas"],
    )
    log.info("Saúde: %d marcas", fusion["saude"]["marcas"])
    log.info("Relatórios: relatorio_integrado.json, relatorio_integrado.txt")


if __name__ == "__main__":
    main()
