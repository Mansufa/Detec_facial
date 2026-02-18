import cv2
import numpy as np
from datetime import datetime
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("AVISO: MediaPipe não disponível. Usando detecção facial alternativa.")


class VideoAnalyzer:
    """Análise de vídeos para detectar sinais de depressão, hematomas e problemas de saúde"""

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        # Configuração do detector facial
        try:
            # Tenta usar MediaPipe
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_mediapipe = True
        except:
            # Fallback para Haar Cascade do OpenCV
            print("Usando detector facial alternativo (Haar Cascade)")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            self.use_mediapipe = False
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

                def analyze_facial_expression(self, landmarks, frame_shape):
                """Analisa expressões faciais para detectar sinais de depressão"""
                h, w = frame_shape[:2]

                # Pontos chave para análise de expressão
                # Olhos (para detectar cansaço/tristeza)
                left_eye_top = landmarks[159]
                left_eye_bottom = landmarks[145]
                right_eye_top = landmarks[386]
                right_eye_bottom = landmarks[374]

                # Sobrancelhas (para detectar expressão triste)
                left_eyebrow = landmarks[70]
                right_eyebrow = landmarks[300]

                # Boca (para detectar falta de sorriso/tristeza)
                mouth_left = landmarks[61]
                mouth_right = landmarks[291]
                mouth_top = landmarks[13]
                mouth_bottom = landmarks[14]

                # Cálculo de métricas
                eye_openness_left = abs(left_eye_top.y - left_eye_bottom.y) * h
                eye_openness_right = abs(
                    right_eye_top.y - right_eye_bottom.y) * h
                avg_eye_openness = (eye_openness_left + eye_openness_right) / 2

                mouth_width = abs(mouth_left.x - mouth_right.x) * w
                mouth_height = abs(mouth_top.y - mouth_bottom.y) * h
                mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

                # Análise de expressão
                expression_data = {
                    'eye_openness': avg_eye_openness,
                    'mouth_ratio': mouth_ratio,
                    'timestamp': self.results['frames_analisados']
                    }

                # Indicadores de depressão
                indicators = []
                depression_score = 0

                # Olhos pouco abertos (cansaço, falta de energia)
                if avg_eye_openness < 8:
            indicators.append('Olhos com aparência cansada')
            depression_score += 2

                # Boca neutra ou para baixo (falta de sorriso)
                if mouth_ratio < 0.08:
            indicators.append('Expressão facial neutra/triste')
            depression_score += 2

                return expression_data, indicators, depression_score

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

                # Conversão para diferentes espaços de cor
                hsv = cv2.cvtColor(face_area, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(face_area, cv2.COLOR_BGR2LAB)

                bruises = []
                marks = []

                # Detecção de hematomas (tons roxos, azuis escuros, amarelados)
                # Hematomas frescos (roxo/azulado)
                lower_purple = np.array([120, 30, 30])
                upper_purple = np.array([160, 255, 200])
                mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

                # Hematomas mais antigos (amarelado/esverdeado)
                lower_yellow = np.array([20, 40, 40])
                upper_yellow = np.array([40, 255, 200])
                mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

                # Hematomas escuros
                lower_dark = np.array([0, 0, 0])
                upper_dark = np.array([180, 255, 80])
                mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

                # Combina máscaras
                mask_bruise = cv2.bitwise_or(mask_purple, mask_yellow)
                mask_bruise = cv2.bitwise_or(mask_bruise, mask_dark)

                # Remove ruído
                kernel = np.ones((5, 5), np.uint8)
                mask_bruise = cv2.morphologyEx(
                    mask_bruise, cv2.MORPH_OPEN, kernel)
                mask_bruise = cv2.morphologyEx(
                    mask_bruise, cv2.MORPH_CLOSE, kernel)

                # Detecta contornos
                contours, _ = cv2.findContours(
            mask_bruise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
            area = cv2.contourArea(contour)
            # Filtra áreas muito pequenas (ruído) ou muito grandes (sombras)
            if 100 < area < 5000:
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)

                # Calcula localização relativa
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

                    # Detecção de marcas vermelhas (possíveis ferimentos, irritações)
                    lower_red1= np.array([0, 50, 50])
                    upper_red1= np.array([10, 255, 255])
                    lower_red2= np.array([170, 50, 50])
                    upper_red2= np.array([180, 255, 255])

                    mask_red1= cv2.inRange(hsv, lower_red1, upper_red1)
                    mask_red2= cv2.inRange(hsv, lower_red2, upper_red2)
                    mask_red= cv2.bitwise_or(mask_red1, mask_red2)

                    mask_red= cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
                    contours_red, _= cv2.findContours(
                    mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours_red:
            area= cv2.contourArea(contour)
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

                    # Horizontal
                    if rel_x < 0.35:
                    location.append('esquerda')
                    elif rel_x > 0.65:
                    location.append('direita')
                    else:
                    location.append('centro')

                    # Vertical
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

            # Processa apenas alguns frames para otimizar
            if frame_count % sample_rate != 0:
                continue

            processed_count += 1
            self.results['frames_analisados'] = processed_count

            # Converte para RGB para o MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detecta face
            results_face = self.face_mesh.process(rgb_frame)

            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    # Análise de expressão facial
                    expression_data, indicators, depression_score = self.analyze_facial_expression(
                        face_landmarks.landmark, frame.shape
                    )

                    self.results['depressao']['expressoes_detectadas'].append(
                        expression_data)
                    self.results['depressao']['score_depressao'] += depression_score
                    if indicators:
                        self.results['depressao']['indicadores'].extend(
                            indicators)

                    # Calcula bounding box da face
                    h, w = frame.shape[:2]
                    x_coords = [landmark.x *
                               w for landmark in face_landmarks.landmark]
                    y_coords = [landmark.y *
                               h for landmark in face_landmarks.landmark]

                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    face_region = (x_min, y_min, x_max - x_min, y_max - y_min)

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

                    # Processa resultados finais
                    self._process_final_results()

                    return self.results

                    def _process_final_results(self):
                    """Processa e sumariza os resultados finais"""
                    # Média do score de depressão
                    if self.results['frames_analisados'] > 0:
            self.results['depressao']['score_depressao'] /= self.results['frames_analisados']

                    # Remove indicadores duplicados
        self.results['depressao']['indicadores'] = list(set(
            self.results['depressao']['indicadores']
        ))

            # Agrupa hematomas por localização
        location_count = defaultdict(int)
            for bruise in self.results['hematomas']['detectados']:
            location_count[bruise['location']] += 1

        self.results['hematomas']['localizacoes'] = dict(location_count)

            # Agrupa marcas por tipo
            mark_types= defaultdict(int)
            for mark in self.results['marcas']['detectadas']:
            mark_types[mark['type']] += 1

        self.results['marcas']['tipos'] = dict(mark_types)

            def generate_report(self, output_path='analysis_report.json'):
            """Gera relatório completo da análise"""
            # Interpretação dos resultados
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

                # Salva relatório
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4, ensure_ascii=False)

                # Gera relatório em texto
                self._generate_text_report(
                report, output_path.replace('.json', '.txt'))

                return report

                def _interpret_depression_score(self, score):
            """Interpreta o score de depressão"""
            if score < 0.5:
                return 'Baixo - Sem sinais significativos'
                elif score < 1.5:
                return 'Moderado - Alguns indicadores presentes'
                else:
                return 'Alto - Múltiplos indicadores presentes'

                def _get_depression_recommendation(self, score):
            """Retorna recomendação baseada no score de depressão"""
            if score < 0.5:
                return 'Não foram detectados sinais significativos de depressão nas expressões faciais.'
                elif score < 1.5:
                return 'Alguns indicadores de expressão facial podem sugerir cansaço ou tristeza. Recomenda-se observação e diálogo aberto.'
                else:
                return 'ATENÇÃO: Múltiplos indicadores detectados. Recomenda-se fortemente buscar avaliação profissional de saúde mental.'

                def _interpret_bruise_risk(self, score):
            """Interpreta o score de risco de hematomas"""
            if score < 5:
                return 'Baixo - Poucos ou nenhum hematoma detectado'
                elif score < 15:
                return 'Moderado - Alguns hematomas detectados'
                else:
                return 'ALTO - Múltiplos hematomas detectados'

                def _get_bruise_recommendation(self, score, locations):
            """Retorna recomendação baseada nos hematomas"""
            if score < 5:
                return 'Não foram detectados hematomas significativos.'
                elif score < 15:
                rec = 'Foram detectados alguns hematomas. '
                if locations:
                rec += f'Localizações: {", ".join(locations.keys())}. '
                rec += 'Recomenda-se investigar a origem dessas marcas.'
                return rec
                else:
                return f'ALERTA: Múltiplos hematomas detectados em diversas regiões. Localizações: {", ".join(locations.keys())}. RECOMENDAÇÃO URGENTE: Avaliação médica e/ou avaliação de segurança pessoal. Em caso de violência doméstica, ligue 180 (Central de Atendimento à Mulher).'

                def _get_marks_recommendation(self, count):
            """Retorna recomendação baseada nas marcas"""
            if count < 3:
                return 'Poucas ou nenhuma marca detectada.'
                elif count < 8:
                return 'Algumas marcas vermelhas foram detectadas. Podem ser irritações cutâneas, arranhões ou outros problemas de pele. Recomenda-se observação.'
                else:
                return 'Múltiplas marcas detectadas. Recomenda-se avaliação dermatológica ou médica para investigar possíveis problemas de saúde da pele.'

                def _generate_text_report(self, report, output_path):
            """Gera relatório em formato texto legível"""
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
            """Função principal para executar a análise"""
            video_path = 'data/YTDown.com_YouTube_Media_5t_FoFzVcsA_001_720p.mp4'

            if not os.path.exists(video_path):
        print(f"ERRO: Vídeo não encontrado em {video_path}")
            return

            print("="*80)
                print("SISTEMA DE ANÁLISE DE VÍDEO")
                print("Detecção de Sinais de Depressão, Hematomas e Problemas de Saúde")
                print("="*80)
                print()

                # Cria analisador
                analyzer = VideoAnalyzer(video_path)

                # Executa análise
                print("Iniciando análise do vídeo...")
                results = analyzer.analyze_video(sample_rate=30)

                print("\nAnálise concluída!")
                print(f"Total de frames analisados: {results['frames_analisados']}")

                # Gera relatório
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
