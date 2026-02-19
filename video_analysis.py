import json
import logging
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)

YOLO_MODEL = "yolov8n.pt"
PERSON_CLASS_ID = 0

HSV_BRUISE_RANGES = [
    ("hematoma_roxo", np.array([120, 50, 50]), np.array([155, 255, 180])),
    ("hematoma_amarelo", np.array([22, 60, 60]), np.array([38, 200, 180])),
]

HSV_RED_RANGES = [
    (np.array([0, 130, 100]), np.array([6, 255, 200])),
    (np.array([174, 130, 100]), np.array([180, 255, 200])),
]

MORPH_KERNEL = np.ones((5, 5), np.uint8)


def _face_location_label(rel_x: float, rel_y: float) -> str:
    h_label = "esquerda" if rel_x < 0.35 else ("direita" if rel_x > 0.65 else "centro")
    v_label = "testa/superior" if rel_y < 0.33 else ("meio" if rel_y < 0.66 else "inferior/queixo")
    return f"{h_label} - {v_label}"


class VideoAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir: {video_path}")

        self.yolo = YOLO(YOLO_MODEL)
        log.info("YOLOv8 carregado (%s)", YOLO_MODEL)

        self.use_mediapipe = False
        try:
            import mediapipe as mp

            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.use_mediapipe = True
            log.info("MediaPipe FaceMesh disponível")
        except Exception:
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            eye_cascade = cv2.data.haarcascades + "haarcascade_eye.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade)
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade)
            log.warning("MediaPipe indisponível — fallback Haar Cascade")

        self.results = {
            "depressao": {"expressoes": [], "score": 0, "indicadores": []},
            "hematomas": {"detectados": [], "localizacoes": {}, "score_risco": 0},
            "marcas": {"detectadas": [], "tipos": {}},
            "frames_analisados": 0,
            "timestamp": datetime.now().isoformat(),
        }

    def _expr_landmarks(self, landmarks, shape):
        h, w = shape[:2]
        eye_l = abs(landmarks[159].y - landmarks[145].y) * h
        eye_r = abs(landmarks[386].y - landmarks[374].y) * h
        avg_eye = (eye_l + eye_r) / 2

        mouth_w = abs(landmarks[61].x - landmarks[291].x) * w
        mouth_h = abs(landmarks[13].y - landmarks[14].y) * h
        mouth_ratio = mouth_h / mouth_w if mouth_w > 0 else 0

        indicators, score = [], 0
        if avg_eye < 8:
            indicators.append("Olhos semicerrados (cansaço/tristeza)")
            score += 2
        if mouth_ratio < 0.08:
            indicators.append("Expressão neutra/triste (sem sorriso)")
            score += 2

        return {"eye_openness": avg_eye, "mouth_ratio": mouth_ratio}, indicators, score

    def _expr_haar(self, gray_face, frame, x, y, w, h):
        eyes = self.eye_cascade.detectMultiScale(gray_face)
        indicators, score = [], 0

        if len(eyes) < 2:
            indicators.append("Olhos semicerrados (cansaço/tristeza)")
            score += 2

        lower = frame[y + int(h * 0.6) : y + h, x : x + w]
        if lower.size > 0 and np.mean(cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)) < 100:
            indicators.append("Expressão neutra/triste (sem sorriso)")
            score += 1

        return {"eye_openness": float(len(eyes)), "mouth_ratio": 0.0}, indicators, score

    def _detect_skin_anomalies(self, frame, region):
        x, y, w, h = region
        margin = int(h * 0.3)
        y1, y2 = max(0, y - margin), min(frame.shape[0], y + h + margin)
        x1, x2 = max(0, x - margin), min(frame.shape[1], x + w + margin)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return [], []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bruises, marks = [], []

        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for _, lo, hi in HSV_BRUISE_RANGES:
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, MORPH_KERNEL)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, MORPH_KERNEL)

        for cnt in cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(cnt)
            if 300 < area < 5000:
                xc, yc, wc, hc = cv2.boundingRect(cnt)
                loc = _face_location_label((xc + wc / 2) / roi.shape[1], (yc + hc / 2) / roi.shape[0])
                bruises.append({"area": area, "location": loc, "type": "hematoma"})

        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in HSV_RED_RANGES:
            red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv, lo, hi))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, MORPH_KERNEL)

        for cnt in cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(cnt)
            if 500 < area < 3000:
                xc, yc, wc, hc = cv2.boundingRect(cnt)
                loc = _face_location_label((xc + wc / 2) / roi.shape[1], (yc + hc / 2) / roi.shape[0])
                marks.append({"area": area, "location": loc, "type": "marca_vermelha"})

        return bruises, marks

    def analyze(self, sample_rate: int = 30) -> dict:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log.info("Vídeo: %d frames, %.1f FPS", total, fps)

        frame_idx, processed = 0, 0

        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue

            processed += 1
            self.results["frames_analisados"] = processed

            dets = self.yolo(frame, verbose=False)[0]
            persons = [b for b in dets.boxes if int(b.cls[0]) == PERSON_CLASS_ID and float(b.conf[0]) > 0.5]
            if not persons:
                continue

            if self.use_mediapipe:
                self._run_mediapipe(frame)
            else:
                self._run_haar(frame)

            if processed % 10 == 0:
                log.info("Frames processados: %d", processed)

        self.cap.release()
        self._finalize()
        return self.results

    def _run_mediapipe(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return
        for fl in result.multi_face_landmarks:
            expr, inds, score = self._expr_landmarks(fl.landmark, frame.shape)
            self._save_expr(expr, inds, score)
            h, w = frame.shape[:2]
            xs = [lm.x * w for lm in fl.landmark]
            ys = [lm.y * h for lm in fl.landmark]
            region = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            self._save_anomalies(frame, region)

    def _run_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        for x, y, w, h in faces:
            expr, inds, score = self._expr_haar(gray[y : y + h, x : x + w], frame, x, y, w, h)
            self._save_expr(expr, inds, score)
            self._save_anomalies(frame, (x, y, w, h))

    def _save_expr(self, data, indicators, score):
        data["frame"] = self.results["frames_analisados"]
        self.results["depressao"]["expressoes"].append(data)
        self.results["depressao"]["score"] += score
        self.results["depressao"]["indicadores"].extend(indicators)

    def _save_anomalies(self, frame, region):
        bruises, marks = self._detect_skin_anomalies(frame, region)
        if bruises:
            self.results["hematomas"]["detectados"].extend(bruises)
            self.results["hematomas"]["score_risco"] += len(bruises) * 3
        if marks:
            self.results["marcas"]["detectadas"].extend(marks)

    def _finalize(self):
        n = self.results["frames_analisados"]
        if n > 0:
            self.results["depressao"]["score"] /= n
        self.results["depressao"]["indicadores"] = list(set(self.results["depressao"]["indicadores"]))

        loc = defaultdict(int)
        for b in self.results["hematomas"]["detectados"]:
            loc[b["location"]] += 1
        self.results["hematomas"]["localizacoes"] = dict(loc)

        tc = defaultdict(int)
        for m in self.results["marcas"]["detectadas"]:
            tc[m["type"]] += 1
        self.results["marcas"]["tipos"] = dict(tc)

    def generate_report(self, output_path: str = "relatorio_video.json") -> dict:
        r = self.results
        report = {
            "arquivo": self.video_path,
            "timestamp": r["timestamp"],
            "frames_analisados": r["frames_analisados"],
            "depressao": {
                "score": round(r["depressao"]["score"], 2),
                "nivel": self._nivel_depressao(r["depressao"]["score"]),
                "indicadores": r["depressao"]["indicadores"],
            },
            "hematomas": {
                "total": len(r["hematomas"]["detectados"]),
                "score_risco": r["hematomas"]["score_risco"],
                "nivel_risco": self._nivel_hematoma(r["hematomas"]["score_risco"]),
                "localizacoes": r["hematomas"]["localizacoes"],
            },
            "marcas": {
                "total": len(r["marcas"]["detectadas"]),
                "tipos": r["marcas"]["tipos"],
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self._write_txt(report, output_path.replace(".json", ".txt"))
        log.info("Relatórios: %s", output_path)
        return report

    @staticmethod
    def _nivel_depressao(score):
        if score < 0.5:
            return "Baixo"
        return "Moderado" if score < 1.5 else "Alto"

    @staticmethod
    def _nivel_hematoma(score):
        if score < 5:
            return "Baixo"
        return "Moderado" if score < 15 else "Alto"

    @staticmethod
    def _write_txt(report, path):
        sep = "=" * 72
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{sep}\nRELATÓRIO DE ANÁLISE DE VÍDEO\n{sep}\n\n")
            f.write(f"Arquivo: {report['arquivo']}\nData: {report['timestamp']}\n")
            f.write(f"Frames: {report['frames_analisados']}\n\n")

            d = report["depressao"]
            f.write(f"--- DEPRESSÃO ---\nScore: {d['score']}  Nível: {d['nivel']}\n")
            for i in d["indicadores"]:
                f.write(f"  • {i}\n")

            h = report["hematomas"]
            f.write(f"\n--- HEMATOMAS ---\nTotal: {h['total']}  Risco: {h['nivel_risco']}\n")
            for loc, n in h["localizacoes"].items():
                f.write(f"  • {loc}: {n}x\n")

            m = report["marcas"]
            f.write(f"\n--- MARCAS ---\nTotal: {m['total']}\n")
            for t, n in m["tipos"].items():
                f.write(f"  • {t}: {n}x\n")

            f.write(f"\n{sep}\nTriagem automatizada — não substitui avaliação profissional.\n{sep}\n")
