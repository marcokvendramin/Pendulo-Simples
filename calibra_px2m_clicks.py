"""
calibra_px2m_clicks.py
Uso:
  python calibra_px2m_clicks.py --image frame_calibracao.png --length_m 0.10 --out_json px2m.json --out_overlay calib_overlay.png

- Abra a imagem, CLIQUE em duas bordas do segmento de comprimento real conhecido.
- O script calcula:
    px2m = length_m / Npx   (m por pixel)
"""

import argparse, json, math
from pathlib import Path
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Imagem com a régua/objeto")
    ap.add_argument("--length_m", type=float, required=True, help="Comprimento REAL entre os dois cliques (em metros)")
    ap.add_argument("--out_json", default="px2m.json", help="Arquivo de saída com o valor (opcional)")
    ap.add_argument("--out_overlay", default="calib_overlay.png", help="Imagem com linha desenhada (opcional)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Não consegui abrir a imagem: {args.image}")

    pts = []
    win = "Clique 2 pontos (ESC para sair)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(900, img.shape[1]), min(1200, img.shape[0]))

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(win, img)

    cv2.imshow(win, img)
    cv2.setMouseCallback(win, on_click)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        if len(pts) >= 2:
            break

    cv2.destroyAllWindows()

    if len(pts) < 2:
        raise SystemExit("Seleção incompleta: clique dois pontos no trecho de comprimento conhecido.")

    (x1, y1), (x2, y2) = pts[:2]
    npx = math.hypot(x2 - x1, y2 - y1)
    if npx <= 0:
        raise SystemExit("Distância em pixels nula. Verifique os cliques.")

    px2m = args.length_m / npx

    # Overlay com a linha medido
    img2 = cv2.imread(args.image)
    cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.circle(img2, (x1, y1), 5, (255, 0, 0), -1)
    cv2.circle(img2, (x2, y2), 5, (255, 0, 0), -1)
    label = f"Npx={npx:.2f} | L={args.length_m} m | px2m={px2m:.9f} m/px"
    cv2.rectangle(img2, (10, 10), (10 + 8*len(label), 45), (255, 255, 255), -1)
    cv2.putText(img2, label, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    cv2.imwrite(args.out_overlay, img2)

    data = {"px2m": px2m, "Npx": npx, "L_real_m": args.length_m}
    Path(args.out_json).write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"px2m = {px2m:.9f} m/px")
    print(f"Npx  = {npx:.2f} px")
    print(f"Overlay salvo em: {args.out_overlay}")
    print(f"JSON salvo em    : {args.out_json}")

if __name__ == "__main__":
    main()
