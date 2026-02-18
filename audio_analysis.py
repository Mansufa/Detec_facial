import os
import subprocess
import json
from pathlib import Path
import re


class AudioAnalyzer:
    """Análise de áudio para detectar sinais de depressão na fala"""

    def __init__(self, video_path):
        self.video_path = video_path
        self.audio_path = None
        self.results = {
            'transcricao': '',
            'palavras_chave_depressao': [],
            'score_depressao_fala': 0,
            'indicadores_linguisticos': [],
            'caracteristicas_voz': {}
        }

        # Palavras e frases indicadoras de depressão
        self.depression_keywords = [
            # Sentimentos negativos
            'triste', 'tristeza', 'deprimido', 'deprimida', 'deprimente',
            'sozinho', 'sozinha', 'solidão', 'vazio', 'vazia',
            'desesperado', 'desesperada', 'sem esperança', 'desespero',
            'cansado', 'cansada', 'exausto', 'exausta', 'esgotado', 'esgotada',

            # Pensamentos negativos
            'não consigo', 'não aguento', 'não dá mais',
            'sem sentido', 'sem propósito', 'inútil',
            'fracasso', 'fracassado', 'fracassada',
            'culpa', 'culpado', 'culpada',

            # Isolamento
            'ninguém entende', 'ninguém se importa', 'sozinho no mundo',
            'me afastar', 'isolar', 'isolamento',

            # Sintomas físicos
            'não durmo', 'insônia', 'não como', 'sem apetite',
            'dor', 'corpo pesado', 'sem energia',

            # Ideação
            'desistir', 'acabar com tudo', 'sumir',

            # Emoções
            'angústia', 'ansiedade', 'medo', 'pavor',
            'choro', 'chorando', 'chorar'
        ]

        # Padrões linguísticos
        self.negative_patterns = [
            r'\bnão\s+\w+',  # negações
            r'\bnunca\b',
            r'\bnada\b',
            r'\bsempre\s+(triste|mal|cansado|sozinho)',
        ]

    def extract_audio(self):
        """Extrai áudio do vídeo"""
        try:
            audio_path = self.video_path.replace('.mp4', '_audio.wav')

            # Usa ffmpeg para extrair áudio
            command = [
                'ffmpeg',
                '-i', self.video_path,
                '-vn',  # sem vídeo
                '-acodec', 'pcm_s16le',  # codec de áudio
                '-ar', '16000',  # taxa de amostragem
                '-ac', '1',  # mono
                '-y',  # sobrescreve
                audio_path
            ]

            subprocess.run(command, check=True, capture_output=True)
            self.audio_path = audio_path
            print(f"Áudio extraído: {audio_path}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Erro ao extrair áudio: {e}")
            return False
        except FileNotFoundError:
            print("AVISO: ffmpeg não encontrado. Instalando dependências necessárias...")
            print(
                "Por favor, instale o ffmpeg manualmente ou use: pip install imageio-ffmpeg")
            return False

    def transcribe_audio(self):
        """Transcreve o áudio usando speech recognition"""
        if not self.audio_path or not os.path.exists(self.audio_path):
            print("Áudio não encontrado. Extraindo áudio primeiro...")
            if not self.extract_audio():
                return ""

        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()

            # Lê o arquivo de áudio
            with sr.AudioFile(self.audio_path) as source:
                print("Carregando áudio...")
                audio = recognizer.record(source)

            print("Transcrevendo áudio (isso pode levar alguns minutos)...")

            # Tenta transcrever usando o Google Speech Recognition (gratuito)
            try:
                text = recognizer.recognize_google(audio, language='pt-BR')
                self.results['transcricao'] = text
                print("Transcrição concluída!")
                return text
            except sr.UnknownValueError:
                print("Não foi possível entender o áudio.")
                return ""
            except sr.RequestError as e:
                print(f"Erro no serviço de reconhecimento: {e}")
                return ""

        except ImportError:
            print("AVISO: SpeechRecognition não instalado.")
            print("Para análise de áudio, instale: pip install SpeechRecognition")
            return ""
        except Exception as e:
            print(f"Erro na transcrição: {e}")
            return ""

    def analyze_text_for_depression(self, text):
        """Analisa o texto transcrito para sinais de depressão"""
        if not text:
            return

        text_lower = text.lower()
        score = 0
        found_keywords = []
        indicators = []

        # Procura palavras-chave
        for keyword in self.depression_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
                score += 2

        # Analisa padrões linguísticos
        for pattern in self.negative_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                indicators.append(f"Padrão negativo: {pattern}")
                score += len(matches)

        # Analisa tom geral
        negative_words = ['não', 'nunca', 'nada', 'nenhum', 'nem']
        negative_count = sum(text_lower.count(word) for word in negative_words)

        if negative_count > 5:
            indicators.append("Alto uso de palavras negativas")
            score += negative_count * 0.5

        # Analisa primeira pessoa (foco em si mesmo)
        first_person = ['eu', 'me', 'meu', 'minha', 'mim']
        first_person_count = sum(text_lower.count(word)
                                 for word in first_person)

        if first_person_count > 10:
            indicators.append(
                "Foco excessivo em si mesmo (possível ruminação)")
            score += 2

        self.results['palavras_chave_depressao'] = found_keywords
        self.results['score_depressao_fala'] = score
        self.results['indicadores_linguisticos'] = indicators

    def analyze_audio_features(self):
        """Analisa características vocais (tom, velocidade, etc.)"""
        # Esta funcionalidade requer bibliotecas mais avançadas como librosa
        # Por enquanto, retorna um placeholder
        try:
            import librosa
            import numpy as np

            if not self.audio_path or not os.path.exists(self.audio_path):
                return

            # Carrega áudio
            y, sr = librosa.load(self.audio_path, sr=16000)

            # Analisa características
            # Pitch (tom)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0])

            # Energia
            energy = np.sum(librosa.feature.rms(y=y))

            # Taxa de zero crossing (pode indicar qualidade/emoção da voz)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))

            self.results['caracteristicas_voz'] = {
                'pitch_medio': float(pitch_mean) if not np.isnan(pitch_mean) else 0,
                'energia': float(energy),
                'zero_crossing_rate': float(zcr)
            }

            # Interpretação
            if pitch_mean < 120:
                self.results['indicadores_linguisticos'].append(
                    "Tom de voz baixo (pode indicar baixa energia/tristeza)"
                )
                self.results['score_depressao_fala'] += 1

            if energy < 100:
                self.results['indicadores_linguisticos'].append(
                    "Baixa energia vocal"
                )
                self.results['score_depressao_fala'] += 1

        except ImportError:
            print("AVISO: librosa não instalado. Análise vocal avançada desabilitada.")
            print("Para análise vocal, instale: pip install librosa")
        except Exception as e:
            print(f"Erro na análise de características vocais: {e}")

    def analyze(self, transcription_text=None):
        """Executa análise completa do áudio"""
        print("\n" + "="*80)
        print("ANÁLISE DE ÁUDIO")
        print("="*80)

        # Se transcrição não foi fornecida, tenta transcrever
        if transcription_text is None:
            # Extrai áudio
            if self.extract_audio():
                # Transcreve
                transcription_text = self.transcribe_audio()
        else:
            self.results['transcricao'] = transcription_text

        # Analisa texto
        if transcription_text:
            print("\nAnalisando conteúdo da fala...")
            self.analyze_text_for_depression(transcription_text)

            # Analisa características vocais
            print("Analisando características vocais...")
            self.analyze_audio_features()
        else:
            print("Sem transcrição disponível para análise.")

        return self.results

    def generate_report(self, output_path='audio_analysis_report.json'):
        """Gera relatório da análise de áudio"""
        report = {
            'arquivo_analisado': self.video_path,
            'transcricao': self.results['transcricao'],
            'analise_fala': {
                'score_depressao': self.results['score_depressao_fala'],
                'nivel': self._interpret_speech_score(
                    self.results['score_depressao_fala']
                ),
                'palavras_chave_encontradas': self.results['palavras_chave_depressao'],
                'indicadores_linguisticos': self.results['indicadores_linguisticos'],
                'caracteristicas_voz': self.results['caracteristicas_voz'],
                'recomendacao': self._get_speech_recommendation(
                    self.results['score_depressao_fala']
                )
            }
        }

        # Salva JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        # Gera texto
        self._generate_text_report(
            report, output_path.replace('.json', '.txt'))

        return report

    def _interpret_speech_score(self, score):
        """Interpreta o score de depressão na fala"""
        if score < 5:
            return 'Baixo - Poucos indicadores na fala'
        elif score < 15:
            return 'Moderado - Alguns indicadores presentes'
        else:
            return 'Alto - Múltiplos indicadores de depressão na fala'

    def _get_speech_recommendation(self, score):
        """Retorna recomendação baseada na análise de fala"""
        if score < 5:
            return 'Não foram detectados sinais significativos de depressão na fala.'
        elif score < 15:
            return 'Alguns indicadores linguísticos sugerem possível tristeza ou desânimo. Recomenda-se atenção e diálogo.'
        else:
            return 'ATENÇÃO: Múltiplos indicadores de depressão detectados na fala. Recomenda-se URGENTEMENTE avaliação profissional de saúde mental. CVV: 188 (24h)'

    def _generate_text_report(self, report, output_path):
        """Gera relatório em texto"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DE ÁUDIO/FALA\n")
            f.write("="*80 + "\n\n")

            f.write(f"Arquivo: {report['arquivo_analisado']}\n\n")

            f.write("-"*80 + "\n")
            f.write("TRANSCRIÇÃO\n")
            f.write("-"*80 + "\n")
            if report['transcricao']:
                f.write(report['transcricao'] + "\n\n")
            else:
                f.write("Transcrição não disponível.\n\n")

            f.write("-"*80 + "\n")
            f.write("ANÁLISE DE INDICADORES DE DEPRESSÃO NA FALA\n")
            f.write("-"*80 + "\n")
            f.write(f"Score: {report['analise_fala']['score_depressao']}\n")
            f.write(f"Nível: {report['analise_fala']['nivel']}\n\n")

            if report['analise_fala']['palavras_chave_encontradas']:
                f.write("Palavras-chave relacionadas à depressão encontradas:\n")
                for palavra in report['analise_fala']['palavras_chave_encontradas'][:10]:
                    f.write(f"  • {palavra}\n")
                if len(report['analise_fala']['palavras_chave_encontradas']) > 10:
                    f.write(
                        f"  ... e mais {len(report['analise_fala']['palavras_chave_encontradas']) - 10}\n")
                f.write("\n")

            if report['analise_fala']['indicadores_linguisticos']:
                f.write("Indicadores Linguísticos:\n")
                for ind in report['analise_fala']['indicadores_linguisticos']:
                    f.write(f"  • {ind}\n")
                f.write("\n")

            if report['analise_fala']['caracteristicas_voz']:
                f.write("Características Vocais:\n")
                for chave, valor in report['analise_fala']['caracteristicas_voz'].items():
                    f.write(f"  • {chave}: {valor:.2f}\n")
                f.write("\n")

            f.write(
                f"Recomendação: {report['analise_fala']['recomendacao']}\n\n")

            f.write("="*80 + "\n")


def main():
    """Função principal"""
    video_path = 'data/YTDown.com_YouTube_Media_5t_FoFzVcsA_001_720p.mp4'

    analyzer = AudioAnalyzer(video_path)
    results = analyzer.analyze()

    if results['transcricao']:
        report = analyzer.generate_report()
        print("\nRelatório de áudio salvo!")
    else:
        print("\nNão foi possível completar a análise de áudio.")


if __name__ == "__main__":
    main()
