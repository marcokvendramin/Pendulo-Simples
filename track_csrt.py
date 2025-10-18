import sys, ctypes, math
from pathlib import Path
import cv2
import pandas as pd

def create_csrt_tracker():
    """Cria um tracker CSRT compatível com OpenCV 4.x (contrib)."""
    try:
        return cv2.legacy.TrackerCSRT_create()
    except Exception:
        try:
            return cv2.TrackerCSRT_create()
        except Exception as e:
            raise RuntimeError(
                "Não consegui criar o Tracker CSRT. "
                "Instale 'opencv-contrib-python' (não headless). Ex.: "
                "pip install --upgrade opencv-contrib-python"
            ) from e

def main():
    if len(sys.argv) < 3:
        print("Uso: python track_csrt.py <video> <saida_dir> [--scale 0.7]")
        sys.exit(2)

    video = sys.argv[1]
    outdir = Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)

    # scale padrão = 1.0; se passar --scale X, usa X como limite superior
    scale_cli = 1.0
    if len(sys.argv) >= 5 and sys.argv[3] == "--scale":
        try:
            scale_cli = float(sys.argv[4])
        except:
            pass

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Não abriu o vídeo: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        raise RuntimeError("Falha ao ler o primeiro frame.")

    # --- AUTO-FIT da janela respeitando o --scale (mostra a imagem inteira) ---
    try:
        sw = ctypes.windll.user32.GetSystemMetrics(0) * 0.95
        sh = ctypes.windll.user32.GetSystemMetrics(1) * 0.95
    except Exception:
        sw, sh = frame0.shape[1], frame0.shape[0]

    auto_scale = min(sw / frame0.shape[1], sh / frame0.shape[0])
    scale = min(scale_cli, auto_scale)
    if scale <= 0:
        scale = 1.0

    # --- Seleção de ROI (com redimensionamento apenas para visualização) ---
    win = "Selecione a massa (ENTER confirma, C cancela)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    Wv = int(frame0.shape[1] * scale)
    Hv = int(frame0.shape[0] * scale)
    disp = frame0 if abs(scale - 1.0) < 1e-6 else cv2.resize(frame0, (Wv, Hv))
    cv2.resizeWindow(win, max(320, min(Wv, 1600)), max(240, min(Hv, 900)))
    cv2.moveWindow(win, 80, 60)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(win, disp)
    cv2.waitKey(50)  # garante render da janela
    bbox_disp = cv2.selectROI(win, disp, False, False)
    cv2.destroyWindow(win)

    x, y, w, h = map(int, bbox_disp)
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("ROI inválida (talvez cancelada com 'c'). Selecione novamente.")

    # Reescala a ROI para o tamanho original
    if abs(scale - 1.0) > 1e-6:
        x = int(x / scale); y = int(y / scale); w = int(w / scale); h = int(h / scale)

    # --- Cria tracker CSRT e inicializa no frame0 ---
    tracker = create_csrt_tracker()
    ok_init = tracker.init(frame0, (x, y, w, h))
    if not ok_init:
        cap.release()
        raise RuntimeError("Falha ao inicializar o tracker CSRT. Tente uma ROI sem fio/sombra.")

    # --- Saída de vídeo (overlay) ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame0.shape[1]
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame0.shape[0]
    outv = cv2.VideoWriter(str(outdir / "overlay.mp4"), fourcc, fps, (W, H))

    # --- Processar desde o frame 0 ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frames, xs, ys = [], [], []
    f = 0

    # Janela de preview
    preview = "Rastreamento (q sai)"
    cv2.namedWindow(preview, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview, min(W, 1280), min(H, 720))
    cv2.moveWindow(preview, 60, 80)
    cv2.setWindowProperty(preview, cv2.WND_PROP_TOPMOST, 1)

    while True:
        okf, frame = cap.read()
        if not okf or frame is None:
            break

        try:
            ok_upd, bb = tracker.update(frame)
        except Exception:
            ok_upd, bb = False, (0, 0, 0, 0)

        cx = cy = float("nan")
        if ok_upd:
            x1, y1, w1, h1 = [int(v) for v in bb]
            cx, cy = x1 + w1 / 2.0, y1 + h1 / 2.0
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)
        else:
            cv2.putText(frame, "ALVO PERDIDO", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Info de FPS e frame
        cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {f}", (20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        outv.write(frame)
        frames.append(f); xs.append(cx); ys.append(cy); f += 1

        cv2.imshow(preview, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyWindow(preview)
    cap.release(); outv.release()

    # Salva CSV
    df = pd.DataFrame({"frame": frames, "x": xs, "y": ys, "fps": fps})
    # Interpola pequenas perdas
    if df["x"].isna().any(): df["x"] = df["x"].interpolate(limit_direction="both")
    if df["y"].isna().any(): df["y"] = df["y"].interpolate(limit_direction="both")
    df["t"] = df["frame"] / fps
    csvpath = outdir / "tracking_xy.csv"
    df.to_csv(csvpath, index=False, encoding="utf-8")

    print("OK:", outdir)
    print("Frames gravados:", len(df), "| CSV:", csvpath)

if __name__ == "__main__":
    main()
