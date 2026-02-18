import cv2
import numpy as np
from datetime import datetime
import json
import os
from collections import defaultdict


class SimpleVideoAnalyzer:
    """Análise simplificada de vídeos para detectar sinais de depressão, hematomas e problemas de saúde"""

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        # Usar detectores Haar Cascade (mais simples e confiável)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

        # Resultados da análise
        self.results = {
            'depressao': {
                'expressoes_detectadas': [],
                'score_depressao': 0,
                'indicadores': []
            },
            'hematomas': {
                'detectados': [],
                'localizacoes': [],
                'score_risco': 0
            },
            'marcas': {
                'detectadas': [],
                'tipos': []
            },
            'frames_analisados': 0,
            'timestamp': datetime.now().isoformat()
        }

    def detect_bruises_and_marks(self, frame, face_region):
        """Detecta hematomas, marcas e possíveis sinais de violência ou problemas de saúde"""
        x, y, w, h = face_region

        # Extrai região da face com margem para pescoço e orelhas
        margin = int(h * 0.3)
        y1 = max(0, y - margin)
        y2 = min(frame.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(frame.shape[1], x + w + margin)

        face_area = frame[y1:y2, x1:x2]

        if face_area.size == 0:
            return [], []

        # Conversão para HSV
        hsv = cv2.cvtColor(face_area, cv2.COLOR_BGR2HSV)

        bruises = []
        marks = []

        # Detecção de hematomas (tons roxos, azuis escuros, amarelados)
        lower_purple = np.array([120, 30, 30])
        upper_purple = np.array([160, 255, 200])
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

        lower_yellow = np.array([20, 40, 40])
        upper_yellow = np.array([40, 255, 200])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

        mask_bruise = cv2.bitwise_or(mask_purple, mask_yellow)
        mask_bruise = cv2.bitwise_or(mask_bruise, mask_dark)

        kernel = np.ones((5, 5), np.uint8)
        mask_bruise = cv2.morphologyEx(mask_bruise, cv2.MORPH_OPEN, kernel)
        mask_bruise = cv2.morphologyEx(mask_bruise, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask_bruise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)

                relative_x = (x_c + w_c/2) / face_area.shape[1]
                relative_y = (y_c + h_c/2) / face_area.shape[0]

                location = self._determine_face_location(
                    relative_x, relative_y)

                bruises.append({
                    'area': area,
                    'location': location,
                    'coords': (x_c, y_c, w_c, h_c),
                    'type': 'hematoma_possivel'
                })

        # Detecção de marcas vermelhas
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        contours_red, _ = cv2.findContours(
            mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_red:
            area = cv2.contourArea(contour)
            if 80 < area < 3000:
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)

                relative_x = (x_c + w_c/2) / face_area.shape[1]
                relative_y = (y_c + h_c/2) / face_area.shape[0]

                location = self._determine_face_location(
                    relative_x, relative_y)

                marks.append({
                    'area': area,
                    'location': location,
                    'coords': (x_c, y_c, w_c, h_c),
                    'type': 'marca_vermelha'
                })

        return bruises, marks

    def _determine_face_location(self, rel_x, rel_y):
        """Determina a localização na face com base em coordenadas relativas"""
        location = []

        if rel_x < 0.35:
            location.append('esquerda')
        elif rel_x > 0.65:
            location.append('direita')
        else:
            location.append('centro')

        if rel_y < 0.33:
            location.append('testa/superior')
        elif rel_y < 0.66:
            location.append('meio')
        else:
            location.append('inferior/queixo')

        return ' - '.join(location)

    def analyze_video(self, sample_rate=30):
        """Analisa o vídeo completo"""
        frame_count = 0
        processed_count = 0

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Iniciando análise do vídeo...")
        print(f"Total de frames: {total_frames}, FPS: {fps}")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % sample_rate != 0:
                continue

            processed_count += 1
            self.results['frames_analisados'] = processed_count

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_region = (x, y, w, h)
                face_roi = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(face_roi)

                # Análise simplificada
                depression_score = 0
                indicators = []

                # Detecta olhos para avaliar cansaço
                if len(eyes) < 2:
                    indicators.append(
                        'Dificuldade em detectar ambos os olhos (possível cansaço ou expressão fechada)')
                    depression_score += 1

                # Análise de brilho (pessoas deprimidas podem ter expressão "apagada")
                face_brightness = np.mean(face_roi)
                if face_brightness < 80:
                    indicators.append(
                        'Expressão com baixa luminosidade (pode indicar rosto "apagado")')
                    depression_score += 1

                expression_data = {
                    'eyes_detected': len(eyes),
                    'face_brightness': float(face_brightness),
                    'timestamp': processed_count
                }

                self.results['depressao']['expressoes_detectadas'].append(
                    expression_data)
                self.results['depressao']['score_depressao'] += depression_score
                if indicators:
                    self.results['depressao']['indicadores'].extend(indicators)

                # Detecção de hematomas e marcas
                bruises, marks = self.detect_bruises_and_marks(
                    frame, face_region)

                if bruises:
                    self.results['hematomas']['detectados'].extend(bruises)
                    self.results['hematomas']['score_risco'] += len(
                        bruises) * 3

                if marks:
                    self.results['marcas']['detectadas'].extend(marks)

            if processed_count % 10 == 0:
                print(f"Processados {processed_count} frames...")

        self.cap.release()
        self._process_final_results()

        return self.results

    def _process_final_results(self):
        """Processa e sumariza os resultados finais"""
        if self.results['frames_analisados'] > 0:
            self.results['depressao']['score_depressao'] /= self.results['frames_analisados']

        self.results['depressao']['indicadores'] = list(set(
            self.results['depressao']['indicadores']
        ))

        location_count = defaultdict(int)
        for bruise in self.results['hematomas']['detectados']:
            location_count[bruise['location']] += 1

        self.results['hematomas']['localizacoes'] = dict(location_count)

        mark_types = defaultdict(int)
        for mark in self.results['marcas']['detectadas']:
            mark_types[mark['type']] += 1

        self.results['marcas']['tipos'] = dict(mark_types)

    def generate_report(self, output_path='analysis_report.json'):
        """Gera relatório completo da análise"""
        report = {
            'arquivo_analisado': self.video_path,
            'timestamp_analise': self.results['timestamp'],
            'frames_analisados': self.results['frames_analisados'],

            'analise_depressao': {
                'score': round(self.results['depressao']['score_depressao'], 2),
                'nivel': self._interpret_depression_score(
                    self.results['depressao']['score_depressao']
                ),
                'indicadores_encontrados': self.results['depressao']['indicadores'],
                'recomendacao': self._get_depression_recommendation(
                    self.results['depressao']['score_depressao']
                )
            },

            'analise_hematomas': {
                'total_detectado': len(self.results['hematomas']['detectados']),
                'score_risco': self.results['hematomas']['score_risco'],
                'nivel_risco': self._interpret_bruise_risk(
                    self.results['hematomas']['score_risco']
                ),
                'localizacoes': self.results['hematomas']['localizacoes'],
                'recomendacao': self._get_bruise_recommendation(
                    self.results['hematomas']['score_risco'],
                    self.results['hematomas']['localizacoes']
                )
            },

            'analise_marcas': {
                'total_detectado': len(self.results['marcas']['detectadas']),
                'tipos': self.results['marcas']['tipos'],
                'recomendacao': self._get_marks_recommendation(
                    len(self.results['marcas']['detectadas'])
                )
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        self._generate_text_report(
            report, output_path.replace('.json', '.txt'))

        return report

    def _interpret_depression_score(self, score):
        if score < 0.5:
            return 'Baixo - Sem sinais significativos'
        elif score < 1.5:
            return 'Moderado - Alguns indicadores presentes'
        else:
            return 'Alto - Múltiplos indicadores presentes'

    def _get_depression_recommendation(self, score):
        if score < 0.5:
            return 'Não foram detectados sinais significativos de depressão nas expressões faciais.'
        elif score < 1.5:
            return 'Alguns indicadores de expressão facial podem sugerir cansaço ou tristeza. Recomenda-se observação e diálogo aberto.'
        else:
            return 'ATENÇÃO: Múltiplos indicadores detectados. Recomenda-se fortemente buscar avaliação profissional de saúde mental.'

    def _interpret_bruise_risk(self, score):
        if score < 5:
            return 'Baixo - Poucos ou nenhum hematoma detectado'
        elif score < 15:
            return 'Moderado - Alguns hematomas detectados'
        else:
            return 'ALTO - Múltiplos hematomas detectados'

    def _get_bruise_recommendation(self, score, locations):
        if score < 5:
            return 'Não foram detectados hematomas significativos.'
        elif score < 15:
            rec = 'Foram detectados alguns hematomas. '
            if locations:
                rec += f'Localizações: {", ".join(locations.keys())}. '
            rec += 'Recomenda-se investigar a origem dessas marcas.'
            return rec
        else:
            return f'ALERTA: Múltiplos hematomas detectados. Localizações: {", ".join(locations.keys())}. RECOMENDAÇÃO URGENTE: Avaliação médica e/ou avaliação de segurança pessoal. Em caso de violência doméstica, ligue 180.'

    def _get_marks_recommendation(self, count):
        if count < 3:
            return 'Poucas ou nenhuma marca detectada.'
        elif count < 8:
            return 'Algumas marcas vermelhas foram detectadas. Podem ser irritações cutâneas, arranhões ou outros problemas de pele. Recomenda-se observação.'
        else:
            return 'Múltiplas marcas detectadas. Recomenda-se avaliação dermatológica ou médica para investigar possíveis problemas de saúde da pele.'

    def _generate_text_report(self, report, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DE VÍDEO\n")
            f.write(
                "Detecção de Sinais de Depressão, Hematomas e Problemas de Saúde\n")
            f.write("="*80 + "\n\n")

            f.write(f"Arquivo Analisado: {report['arquivo_analisado']}\n")
            f.write(f"Data da Análise: {report['timestamp_analise']}\n")
            f.write(f"Frames Analisados: {report['frames_analisados']}\n\n")

            f.write("-"*80 + "\n")
            f.write("1. ANÁLISE DE SINAIS DE DEPRESSÃO (Expressões Faciais)\n")
            f.write("-"*80 + "\n")
            f.write(
                f"Score de Depressão: {report['analise_depressao']['score']}\n")
            f.write(f"Nível: {report['analise_depressao']['nivel']}\n\n")

            if report['analise_depressao']['indicadores_encontrados']:
                f.write("Indicadores Encontrados:\n")
                for ind in report['analise_depressao']['indicadores_encontrados']:
                    f.write(f"  • {ind}\n")
            else:
                f.write("Nenhum indicador significativo encontrado.\n")

            f.write(
                f"\nRecomendação: {report['analise_depressao']['recomendacao']}\n\n")

            f.write("-"*80 + "\n")
            f.write("2. ANÁLISE DE HEMATOMAS (Possível Violência Doméstica)\n")
            f.write("-"*80 + "\n")
            f.write(
                f"Total de Hematomas Detectados: {report['analise_hematomas']['total_detectado']}\n")
            f.write(
                f"Score de Risco: {report['analise_hematomas']['score_risco']}\n")
            f.write(
                f"Nível de Risco: {report['analise_hematomas']['nivel_risco']}\n\n")

            if report['analise_hematomas']['localizacoes']:
                f.write("Localizações dos Hematomas:\n")
                for loc, count in report['analise_hematomas']['localizacoes'].items():
                    f.write(f"  • {loc}: {count} ocorrência(s)\n")
            else:
                f.write("Nenhum hematoma detectado.\n")

            f.write(
                f"\nRecomendação: {report['analise_hematomas']['recomendacao']}\n\n")

            f.write("-"*80 + "\n")
            f.write("3. ANÁLISE DE MARCAS E MACHUCADOS (Problemas de Saúde)\n")
            f.write("-"*80 + "\n")
            f.write(
                f"Total de Marcas Detectadas: {report['analise_marcas']['total_detectado']}\n\n")

            if report['analise_marcas']['tipos']:
                f.write("Tipos de Marcas:\n")
                for tipo, count in report['analise_marcas']['tipos'].items():
                    f.write(f"  • {tipo}: {count} ocorrência(s)\n")
            else:
                f.write("Nenhuma marca significativa detectada.\n")

            f.write(
                f"\nRecomendação: {report['analise_marcas']['recomendacao']}\n\n")

            f.write("="*80 + "\n")
            f.write("IMPORTANTE:\n")
            f.write(
                "Esta análise é baseada em processamento de imagem por computador e deve ser\n")
            f.write(
                "considerada como uma ferramenta de triagem, não um diagnóstico definitivo.\n")
            f.write("Recomenda-se sempre consulta com profissionais qualificados.\n")
            f.write("="*80 + "\n")


def main():
    video_path = 'data/YTDown.com_YouTube_Media_5t_FoFzVcsA_001_720p.mp4'

    if not os.path.exists(video_path):
        print(f"ERRO: Vídeo não encontrado em {video_path}")
        return

    print("="*80)
    print("SISTEMA DE ANÁLISE DE VÍDEO")
    print("Detecção de Sinais de Depressão, Hematomas e Problemas de Saúde")
    print("="*80)
    print()

    analyzer = SimpleVideoAnalyzer(video_path)

    print("Iniciando análise do vídeo...")
    results = analyzer.analyze_video(sample_rate=30)

    print("\nAnálise concluída!")
    print(f"Total de frames analisados: {results['frames_analisados']}")

    print("\nGerando relatórios...")
    report = analyzer.generate_report('analysis_report.json')

    print("\n" + "="*80)
    print("RESUMO DOS RESULTADOS")
    print("="*80)

    print(f"\n1. DEPRESSÃO:")
    print(f"   Score: {report['analise_depressao']['score']}")
    print(f"   Nível: {report['analise_depressao']['nivel']}")

    print(f"\n2. HEMATOMAS:")
    print(
        f"   Total Detectado: {report['analise_hematomas']['total_detectado']}")
    print(f"   Nível de Risco: {report['analise_hematomas']['nivel_risco']}")

    print(f"\n3. MARCAS:")
    print(f"   Total Detectado: {report['analise_marcas']['total_detectado']}")

    print("\n" + "="*80)
    print("Relatórios salvos:")
    print("  • analysis_report.json (formato JSON)")
    print("  • analysis_report.txt (formato texto)")
    print("="*80)


if __name__ == "__main__":
    main()
