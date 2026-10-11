# webcam_yolo_live.py
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="YOLO live webcam with hotkeys & recording")
    p.add_argument("--model", type=str, required=True,
                   help="Chemin des poids (ex: runs/detect/train/weights/best.pt ou .onnx)")
    p.add_argument("--source", type=str, default="0",
                   help="Index webcam (ex: 0) ou chemin vidéo (ex: video.mp4)")
    p.add_argument("--imgsz", type=int, default=640, help="Taille d'entrée (ex: 640)")
    p.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance")
    p.add_argument("--device", type=str, default="cpu", help="cpu, 0 (GPU), 0,1...")
    p.add_argument("--show", action="store_true", help="Afficher la fenêtre")
    p.add_argument("--save", action="store_true", help="Enregistrer la vidéo annotée")
    p.add_argument("--out", type=str, default="runs/live/out.mp4", help="Chemin de sortie vidéo")
    p.add_argument("--stride", type=int, default=1, help="Vid stride (saut d'images) pour alléger la charge")
    return p.parse_args()

def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False

def main():
    args = parse_args()

    print(f"[INFO] Chargement modèle: {args.model}")
    model = YOLO(args.model)

    source_arg = int(args.source) if is_int(args.source) else args.source
    cap = cv2.VideoCapture(source_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    saving = args.save
    if saving:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps_src, (width, height))
        print(f"[INFO] Enregistrement -> {args.out}")

    print(f"[INFO] Classes du modèle: {model.names}")

    print("[INFO] Hotkeys: q=quitter  s=toggle save  p=pause  =/+=conf++  -/=conf--")
    paused = False
    last_time = time.time()
    frame_count = 0
    fps_smoothed = 0.0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Fin du flux ou lecture impossible.")
                    break

                frame_count += 1
                if args.stride > 1 and (frame_count % args.stride != 0):
                    pass

                t0 = time.time()
                results = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=args.device,
                    verbose=False
                )
                dt = time.time() - t0

                annotated = results[0].plot()

                inst_fps = 1.0 / max(dt, 1e-6)
                if fps_smoothed == 0:
                    fps_smoothed = inst_fps
                else:
                    fps_smoothed = 0.9 * fps_smoothed + 0.1 * inst_fps

                cv2.putText(annotated, f"FPS: {fps_smoothed:.1f}  conf={args.conf:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

                if args.show:
                    cv2.imshow("YOLO Live", annotated)

                if saving and writer is not None:
                    writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF if args.show else 0xFF

            if key == ord('q'):
                print("[INFO] Quit.")
                break
            elif key == ord('s'):
                saving = not saving
                if saving and writer is None:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.out, fourcc, fps_src, (width, height))
                    print(f"[INFO] Recording ON -> {args.out}")
                elif not saving and writer is not None:
                    writer.release()
                    writer = None
                    print("[INFO] Recording OFF")
            elif key == ord('p'):
                paused = not paused
                print(f"[INFO] Pause = {paused}")
            elif key in (ord('+'), ord('=')):
                args.conf = min(args.conf + 0.05, 0.95)
            elif key == ord('-'):
                args.conf = max(args.conf - 0.05, 0.05)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()