# webcam_yolo_rectify.py
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CARD_RATIO = 88.0 / 63.0
TARGET_W = 480
TARGET_H = int(TARGET_W * CARD_RATIO)  # ~670

def parse_args():
    p = argparse.ArgumentParser(description="YOLO live webcam with click-to-rectify card")
    p.add_argument("--model", type=str, required=True,
                   help="Poids YOLO (.pt/.onnx) fine-tunés sur 'card'")
    p.add_argument("--source", type=str, default="0",
                   help="Index webcam (0) ou chemin vidéo/URL")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.80)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--show", action="store_true", help="Afficher la fenêtre vidéo")
    p.add_argument("--save_crops_dir", type=str, default="runs/live/crops_rect",
                   help="Dossier de sauvegarde des crops rectifiés (touche 'w')")
    p.add_argument("--stride", type=int, default=1, help="Vid stride")
    return p.parse_args()

def order_points(pts):
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect

def four_point_transform(image, pts, dst_w=TARGET_W, dst_h=TARGET_H):
    rect = order_points(pts.astype(np.float32))
    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC)
    return warped

def find_card_quad(crop_bgr):
    img = crop_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 15, 5)
    edges = cv2.Canny(thr, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w = img.shape[:2]
    img_area = w * h

    quad = None
    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.10:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(-1, 2).astype(np.float32)

            rect = order_points(pts)
            w_est = np.linalg.norm(rect[1] - rect[0])
            h_est = np.linalg.norm(rect[3] - rect[0])
            if w_est == 0 or h_est == 0:
                continue
            ratio = max(w_est, h_est) / max(1.0, min(w_est, h_est))
            if 1.2 <= ratio <= 1.7:
                quad = pts
                break

    dbg = None
    if quad is not None:
        dbg = img.copy()
        cv2.drawContours(dbg, [quad.astype(np.int32)], -1, (0, 255, 0), 2)

    return quad, dbg


class ClickRectify:
    def __init__(self, win_name="YOLO Live", preview_name="Card Preview (rectified)"):
        self.win_name = win_name
        self.preview_name = preview_name
        self.boxes = []
        self.frame_ref = None
        self.last_rect = None

    def update_detections(self, frame, boxes_xyxy, confs, clss):
        self.frame_ref = frame
        self.boxes = []
        for b, c, k in zip(boxes_xyxy, confs, clss):
            x1, y1, x2, y2 = [int(v) for v in b]
            self.boxes.append((x1, y1, x2, y2, float(c), int(k)))

    def _rectify_from_bbox(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        pad = int(0.04 * max(x2 - x1, y2 - y1))
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w - 1, x2 + pad), min(h - 1, y2 + pad)

        crop = frame[y1p:y2p, x1p:x2p].copy()
        if crop.size == 0:
            return None

        quad, _ = find_card_quad(crop)

        if quad is not None:
            rectified = four_point_transform(crop, quad, dst_w=TARGET_W, dst_h=TARGET_H)
            return rectified

        Hc, Wc = crop.shape[:2]
        pts = np.array([[0, 0], [Wc - 1, 0], [Wc - 1, Hc - 1], [0, Hc - 1]], dtype=np.float32)
        rectified = four_point_transform(crop, pts, dst_w=TARGET_W, dst_h=TARGET_H)
        return rectified

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.frame_ref is None or not self.boxes:
            return

        hit = None
        best_conf = -1.0
        for (x1, y1, x2, y2, conf, cls) in self.boxes:
            if x1 <= x <= x2 and y1 <= y <= y2 and conf > best_conf:
                best_conf = conf
                hit = (x1, y1, x2, y2)

        if hit is None:
            return

        rectified = self._rectify_from_bbox(self.frame_ref, hit)
        if rectified is None:
            print("[INFO] Rectification impossible pour ce clic.")
            return

        self.last_rect = rectified
        cv2.imshow(self.preview_name, rectified)

    def save_last_rect(self, out_dir: str):
        if self.last_rect is None:
            print("[INFO] Aucun crop rectifié en mémoire. Clique d'abord sur une carte.")
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        out_path = Path(out_dir) / f"card_rect_{ts}.png"
        cv2.imwrite(str(out_path), self.last_rect)
        print(f"[INFO] Rectifié sauvegardé -> {out_path}")


def is_int(s: str) -> bool:
    try:
        int(s); return True
    except ValueError:
        return False


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
    clicker = ClickRectify(preview_name="Card Preview (rectified)")
    if args.show:
        cv2.setMouseCallback("YOLO Live", clicker.on_mouse)

    print("[INFO] Hotkeys: q=quit  w=save rectified  c=close preview")
    frame_count = 0
    fps_smoothed = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Fin du flux."); break

            frame_count += 1
            if args.stride > 1 and (frame_count % args.stride != 0):
                if args.show:
                    cv2.imshow("YOLO Live", frame)
                key = cv2.waitKey(1) & 0xFF if args.show else 0xFF
                if key == ord('q'):
                    break
                elif key == ord('w'):
                    clicker.save_last_rect(args.save_crops_dir)
                elif key == ord('c'):
                    cv2.destroyWindow("Card Preview (rectified)")
                continue

            t0 = time.time()
            results = model.predict(
                source=frame, imgsz=args.imgsz, conf=args.conf,
                device=args.device, verbose=False
            )
            dt = time.time() - t0
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_smoothed = inst_fps if fps_smoothed == 0 else 0.9*fps_smoothed + 0.1*inst_fps

            r0 = results[0]
            annotated = r0.plot()

            if r0.boxes is not None and len(r0.boxes) > 0:
                boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                clss = r0.boxes.cls.cpu().numpy().astype(int)
            else:
                boxes_xyxy = np.empty((0, 4)); confs = np.empty((0,)); clss = np.empty((0,), dtype=int)
            clicker.update_detections(frame, boxes_xyxy, confs, clss)

            cv2.putText(annotated, f"FPS: {fps_smoothed:.1f} conf={args.conf:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

            if args.show:
                cv2.imshow("YOLO Live", annotated)

            key = cv2.waitKey(1) & 0xFF if args.show else 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                clicker.save_last_rect(args.save_crops_dir)
            elif key == ord('c'):
                cv2.destroyWindow("Card Preview (rectified)")

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()