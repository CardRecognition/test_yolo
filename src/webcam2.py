# webcam_yolo_click_preview.py
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="YOLO live webcam with click-to-preview card")
    p.add_argument("--model", type=str, required=True,
                   help="Poids YOLO (.pt ou .onnx) fine-tunés sur 'card'")
    p.add_argument("--source", type=str, default="0",
                   help="Index webcam (ex: 0) ou chemin vidéo/URL")
    p.add_argument("--imgsz", type=int, default=640, help="Taille entrée YOLO (ex: 640)")
    p.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance")
    p.add_argument("--device", type=str, default="cpu", help="cpu, 0 (GPU), etc.")
    p.add_argument("--show", action="store_true", help="Afficher la fenêtre vidéo")
    p.add_argument("--save_crops_dir", type=str, default="runs/live/crops",
                   help="Dossier pour sauvegarder les crops (touche 'w')")
    p.add_argument("--stride", type=int, default=1, help="Vid stride")
    return p.parse_args()

def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False

class ClickPreview:
    """Gère les clics souris et l’aperçu recadré."""
    def __init__(self, win_name="YOLO Live", preview_name="Card Preview"):
        self.win_name = win_name
        self.preview_name = preview_name
        self.boxes = []
        self.frame_ref = None
        self.last_crop = None

    def update_detections(self, frame, boxes_xyxy, confs, clss):
        self.frame_ref = frame
        self.boxes = []
        for b, c, k in zip(boxes_xyxy, confs, clss):
            x1, y1, x2, y2 = [int(v) for v in b]
            self.boxes.append((x1, y1, x2, y2, float(c), int(k)))

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.frame_ref is None or not self.boxes:
            return

        hit = None
        best_conf = -1.0
        for (x1, y1, x2, y2, conf, cls) in self.boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if conf > best_conf:
                    best_conf = conf
                    hit = (x1, y1, x2, y2, conf, cls)

        if hit is None:
            return

        x1, y1, x2, y2, conf, cls = hit
        h, w = self.frame_ref.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        crop = self.frame_ref[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return

        target_w = 480
        scale = target_w / max(1, crop.shape[1])
        crop_vis = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)),
                              interpolation=cv2.INTER_CUBIC)

        self.last_crop = crop
        cv2.imshow(self.preview_name, crop_vis)
        cv2.setWindowTitle(self.preview_name, f"Card Preview — conf={conf:.2f}")

    def save_last_crop(self, out_dir: str):
        if self.last_crop is None:
            print("[INFO] Aucun crop en mémoire. Clique sur une carte d'abord.")
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        out_path = Path(out_dir) / f"card_{ts}.png"
        cv2.imwrite(str(out_path), self.last_crop)
        print(f"[INFO] Crop sauvegardé -> {out_path}")


def main():
    args = parse_args()

    print(f"[INFO] Chargement modèle: {args.model}")
    model = YOLO(args.model)
    print(f"[INFO] Classes du modèle: {model.names}")

    source_arg = int(args.source) if is_int(args.source) else args.source
    cap = cv2.VideoCapture(source_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source: {args.source}")

    if args.show:
        cv2.namedWindow("YOLO Live", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    clicker = ClickPreview(win_name="YOLO Live", preview_name="Card Preview")
    if args.show:
        cv2.setMouseCallback("YOLO Live", clicker.on_mouse)

    print("[INFO] Hotkeys: q=quit  w=save last crop  c=close preview")
    frame_count = 0
    fps_smoothed = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Fin du flux.")
                break

            frame_count += 1
            if args.stride > 1 and (frame_count % args.stride != 0):
                if args.show:
                    cv2.imshow("YOLO Live", frame)
                key = cv2.waitKey(1) & 0xFF if args.show else 0xFF
                if key == ord('q'):
                    break
                elif key == ord('w'):
                    clicker.save_last_crop(args.save_crops_dir)
                elif key == ord('c'):
                    cv2.destroyWindow("Card Preview")
                continue

            t0 = time.time()
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False
            )
            dt = time.time() - t0
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_smoothed = inst_fps if fps_smoothed == 0 else 0.9*fps_smoothed + 0.1*inst_fps

            r0 = results[0]
            annotated = r0.plot()

            boxes_xyxy = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else np.empty((0, 4))
            confs = r0.boxes.conf.cpu().numpy() if r0.boxes is not None else np.empty((0,))
            clss = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None else np.empty((0,), dtype=int)
            clicker.update_detections(frame, boxes_xyxy, confs, clss)

            cv2.putText(annotated, f"FPS: {fps_smoothed:.1f}  conf={args.conf:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            if args.show:
                cv2.imshow("YOLO Live", annotated)

            key = cv2.waitKey(1) & 0xFF if args.show else 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                clicker.save_last_crop(args.save_crops_dir)
            elif key == ord('c'):
                cv2.destroyWindow("Card Preview")

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()