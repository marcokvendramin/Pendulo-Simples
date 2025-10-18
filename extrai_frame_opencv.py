"""
extrai_frame_opencv.py
Extrai 1 quadro de um vídeo usando OpenCV (sem ffmpeg).

Uso:
  python extrai_frame_opencv.py --video pendulo_simples_60_fps.mp4 --time "00:00:10" --out frame_calibracao.png
ou
  python extrai_frame_opencv.py --video pendulo.mp4 --seconds 10.0 --out frame.png
"""
import argparse, re
import cv2

def parse_hhmmss(s):
    m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?$", s)
    if not m:
        raise ValueError("Formato esperado HH:MM:SS[.ms]")
    h, mi, se, ms = m.groups()
    sec = int(h)*3600 + int(mi)*60 + int(se)
    if ms:
        sec += float("0."+ms)
    return float(sec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--time", help="HH:MM:SS[.ms]")
    g.add_argument("--seconds", type=float, help="segundos desde o início")
    ap.add_argument("--out", default="frame_calibracao.png")
    args = ap.parse_args()

    t = parse_hhmmss(args.time) if args.time else float(args.seconds)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Não abriu vídeo: {args.video}")

    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        # tenta um pequeno recuo (alguns codecs precisam cair num keyframe próximo)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (t-0.2)*1000.0))
        ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise SystemExit("Falha ao capturar o frame solicitado.")

    cv2.imwrite(args.out, frame)
    print(f"Frame salvo em: {args.out}")

if __name__ == "__main__":
    main()
