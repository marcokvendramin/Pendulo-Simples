"""
analisa_pendulo.py

- xy_trajetoria (mesma escala x=y)
- x_vs_t (px) com ajuste + envelopes
- delta_x_vs_t (m) com ajuste + envelopes
- JSONs: ajuste_parametros.json, ajuste_parametros_extras.json, pequenos_angulos.json, incertezas.json

Uso típico:
    python analisa_pendulo_final_min.py --csv res_csrt/tracking_xy.csv --saida res_final ^
        --px2m 0.0005493427612102872 --L 0.720 --dL 0.0005 --m 0.151 --dm 0.0005 --tmax 60
"""

import json, math, argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 120
G_DEFAULT = 9.81

# ------------------------------------------------------------
# MODELO FÍSICO/MATEMÁTICO E MÉTRICAS
# ------------------------------------------------------------
# Modelo do oscilador harmônico amortecido (OHA):
# 
#   x(t) = A * e^{-b t} * cos( ω t + φ ) + C
#
# Onde:
#   A     = amplitude inicial (em pixels)
#   b     = coeficiente de amortecimento (s^-1)
#   ω     = frequência angular amortecida (rad/s)
#   φ     = fase em t=0 (rad)
#   C     = deslocamento médio/offset (px), i.e., centro da oscilação (equilíbrio no eixo da câmera)
#
# Observação: C NÃO é o "zero físico" absoluto; é o centro do movimento no vídeo.

def model_damped_cos(t, A, b, omega, phi, C):
    return A * np.exp(-b * t) * np.cos(omega * t + phi) + C

# Coeficiente de determinação:
#   R² = 1 - (Σ(res²) / Σ((y - ȳ)²))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

# Fator de qualidade (aproximação para amortecimento leve):
#   Q ≈ ω / (2 b)

def compute_quality_factor(b, omega):
    return omega / (2 * b) if b > 0 else np.inf

# ------------------------------------------------------------
# AJUSTE NÃO LINEAR (curve_fit)
# ------------------------------------------------------------
@dataclass
class FitResult:
    A: float; dA: float
    b: float; db: float
    omega: float; domega: float
    phi: float; dphi: float
    C: float; dC: float
    Q: float; f0: float; T0: float
    R2: float

# Chutes iniciais:
#   - ω0 por FFT do sinal detrendido (melhora a convergência)
#   - A0 ~ semi-amplitude dos dados
#   - b0 pequeno (0.01 s^-1), φ0 = 0, C0 = média(x)
def estimate_initial_params(t, x):
    x_detr = x - np.nanmean(x)
    n = len(t); dt = float(np.mean(np.diff(t))) if n > 1 else 0.0
    if n < 8 or dt <= 0:
        omega0 = 2*np.pi*1.0
    else:
        freqs = np.fft.rfftfreq(n, dt)
        X = np.abs(np.fft.rfft(x_detr))
        k = np.argmax(X[1:]) + 1 if len(X) > 1 else 1
        omega0 = 2*np.pi*(freqs[k] if k < len(freqs) and freqs[k] > 0 else 1.0)
    A0 = 0.5 * (np.nanmax(x) - np.nanmin(x))
    return A0, 0.01, omega0, 0.0, float(np.nanmean(x))

def fit_damped_cosine(t, x) -> FitResult:
    # Mascara de dados válidos
    mask = np.isfinite(t) & np.isfinite(x)
    t_fit, x_fit = t[mask], x[mask]

    # Estimativas iniciais e limites físicos razoáveis
    p0 = estimate_initial_params(t_fit, x_fit)
    bounds = ([0,0,0.1,-4*np.pi, np.min(x_fit)-abs(np.mean(x_fit))],
              [np.inf,2.0,100.0, 4*np.pi, np.max(x_fit)+abs(np.mean(x_fit))])

    # Ajuste não linear
    popt, pcov = curve_fit(model_damped_cos, t_fit, x_fit, p0=p0, bounds=bounds, maxfev=200000)
    A,b,omega,phi,C = popt

    # Erros padrão dos parâmetros = sqrt(diag(covariância))
    try:
        dA,db,domega,dphi,dC = np.sqrt(np.diag(pcov))
    except:
        dA=db=domega=dphi=dC = float("nan")

    y_hat = model_damped_cos(t_fit, *popt)
    R2 = r2_score(x_fit, y_hat)

    # Frequência e período:
    #   f0 = ω / (2π)
    #   T0 = 1 / f0
    f0 = omega/(2*np.pi); T0 = 1/f0 if f0>0 else float("nan")

    # Fator de qualidade (aprox.):
    Q = compute_quality_factor(b, omega)

    return FitResult(A,dA,b,db,omega,domega,phi,dphi,C,dC,Q,f0,T0,R2)

# ------------------------------------------------------------
# UTILITÁRIOS DE SAÍDA (JSONs)
# ------------------------------------------------------------
def salvar_parametros_json(outdir: Path, fit: FitResult):
    d = {
        "A": fit.A, "dA": fit.dA,
        "b": fit.b, "db": fit.db,
        "omega": fit.omega, "domega": fit.domega,
        "phi": fit.phi, "dphi": fit.dphi,
        "C": fit.C, "dC": fit.dC,
        "Q": fit.Q, "f0": fit.f0, "T0": fit.T0, "R2": fit.R2,
    }
    (outdir / "ajuste_parametros.json").write_text(json.dumps(d, indent=2), encoding="utf-8")

def salvar_parametros_extras(outdir: Path, fit: FitResult, m: float=None, dm: float=None):
    # Frequência natural amortecida:
    #   ω0 = sqrt( ω^2 + b^2 )
    omega0 = float(np.sqrt(fit.omega**2 + fit.b**2))

    # Período (do ajuste):
    #   T = 2π / ω
    T = 2*np.pi/fit.omega if fit.omega>0 else float("nan")

    # Fator de qualidade “exato” por envelope de ciclos:
    #   Q_exato = 2π / ( 1 - e^{-2 b T} )
    Q_exato = 2*np.pi / (1 - np.exp(-2*fit.b*T)) if (fit.b>0 and T==T) else float("inf")

    extras = {
        "omega0": omega0,
        "T": T,
        "Q_exato": Q_exato,
        # Aproximação alternativa: Q_aprox = ω0 / (2 b)
        "Q_aprox": omega0/(2*fit.b) if fit.b>0 else float("inf")
    }

    # Com massa (opcional), estimamos:
    #   γ = 2 m b   (coeficiente de amortecimento viscoso)
    #   k = m ω0^2  (equivalente à constante elástica efetiva)
    if m is not None:
        gamma = 2*m*fit.b
        k = m*(omega0**2)
        extras.update({"gamma": gamma, "k": k})

    (outdir / "ajuste_parametros_extras.json").write_text(json.dumps(extras, indent=2), encoding="utf-8")

#   Checagem de “pequenos ângulos” (qualitativa):
#   razão Δy/Δx pequena indica movimento majoritariamente horizontal.
#   Se px2m e L forem informados, estimamos θ_max ≈ (Δx/2)*px2m / L.
def pequeno_angulo(df: pd.DataFrame, L_m: float=None, px2m: float=None):
    x = df["x"].values; y = df["y"].values
    rng_x = float(np.nanmax(x) - np.nanmin(x))
    rng_y = float(np.nanmax(y) - np.nanmin(y))
    ratio = rng_y / max(rng_x, 1e-9)
    theta_max = float("nan")
    if L_m and px2m:
        theta_max = (0.5*rng_x * px2m) / L_m
    return {"delta_x_px": rng_x, "delta_y_px": rng_y, "ratio_y_over_x": ratio, "theta_max_rad": theta_max}

def salvar_incertezas(outdir: Path, L: float=None, dL: float=None, m: float=None, dm: float=None, g: float=G_DEFAULT, fit: FitResult=None, px2m: float=None):
    results = {"inputs": {"L": L, "dL": dL, "m": m, "dm": dm, "g": g, "px2m": px2m}}

    #   Propagação para ω_teo = sqrt(g/L):
    #   ∂ω/∂L = - (1/2) sqrt(g) L^{-3/2}
    #   σ_ω ≈ |∂ω/∂L| σ_L   (ignorando σ_g, se não fornecida)
    if L and dL:
        w = math.sqrt(g/L)
        dw = (0.5*math.sqrt(g)*(L**-1.5))*dL
        # T_teo = 2π/ω ; σ_T = (2π/ω^2) σ_ω
        T = 2*math.pi/w; dT = (2*math.pi/w**2)*dw
        results["omega_teo"] = {"value": w, "unc": dw}
        results["T_teo"] = {"value": T, "unc": dT}

    if fit and m and dm:
        omega0 = math.sqrt(fit.omega**2 + fit.b**2)
        # γ = 2 m b ; σ_γ^2 = (2 b σ_m)^2 + (2 m σ_b)^2
        gamma = 2*m*fit.b
        dgamma = math.sqrt((2*fit.b*dm)**2 + (2*m*fit.db)**2) if (fit.db==fit.db) else float("nan")
        # k = m ω0^2 ; dk por diferenciação (aprox. de primeira ordem)
        k = m*(omega0**2)
        domega0 = (1/omega0)*math.sqrt((fit.omega*fit.domega)**2 + (fit.b*fit.db)**2) if (omega0>0 and fit.domega==fit.domega and fit.db==fit.db) else 0.0
        dk = math.sqrt((omega0**2*dm)**2 + (2*m*omega0*domega0)**2) if m is not None else float("nan")
        results["gamma"] = {"value": gamma, "unc": dgamma}
        results["k"] = {"value": k, "unc": dk}

    (outdir / "incertezas.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

# ------------------------------------------------------------
# GRÁFICOS
# ------------------------------------------------------------
def plot_xy_trajetoria(df: pd.DataFrame, outdir: Path):
    # Trajetória (x,y) com escala 1:1 para evidenciar que Δy << Δx
    x = df["x"].values; y = df["y"].values
    dx = float(np.nanmax(x) - np.nanmin(x))
    dy = float(np.nanmax(y) - np.nanmin(y))
    xc = 0.5*(np.nanmax(x)+np.nanmin(x))
    yc = 0.5*(np.nanmax(y)+np.nanmin(y))
    R = 0.55*max(dx, dy)

    plt.figure(figsize=(8.4, 4.4))
    ax = plt.gca()
    ax.plot(x, y, '.', ms=2)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title("Trajetória (x,y) — escala 1:1")
    ax.set_aspect("equal", adjustable="box")
    # Inverter y, pois na imagem a origem costuma ser no topo
    ax.set_xlim(xc-R, xc+R); ax.set_ylim(yc+R, yc-R)
    ax.grid(ls="--", alpha=0.3)
    ax.text(xc, yc-0.9*R, f"Δy/Δx = {dy/max(dx,1e-9):.3f}", ha="center", va="top")
    plt.tight_layout(); plt.savefig(outdir/"xy_trajetoria.png"); plt.close()

def plot_x_vs_t_pixels(df: pd.DataFrame, fit: FitResult, outdir: Path, tmax: float = None):
    # Sinal em pixels “centrado” no C:
    #   delta(t) = x(t) - C
    t = df["t"].values
    x = df["x"].values
    y_hat = model_damped_cos(t, fit.A, fit.b, fit.omega, fit.phi, fit.C)

    # Limitar eixo x até tmax (s), se solicitado
    if tmax:
        mask = t <= tmax
        t, x, y_hat = t[mask], x[mask], y_hat[mask]

    delta = x - fit.C
    delta_fit = y_hat - fit.C

    # Envelopes teóricos do OHA:
    #   +A e^{-b t} e -A e^{-b t}
    env = fit.A*np.exp(-fit.b*t)

    plt.figure(figsize=(8.4, 4.4))
    plt.plot(t, delta, '.', ms=2, label="dados (x − C)")
    plt.plot(t, delta_fit, '-', lw=1.5, label="ajuste OHA")
    plt.plot(t, +env, '--', lw=1, color='red', label="envelope")
    plt.plot(t, -env, '--', lw=1, color='red')
    plt.xlabel("t (s)"); plt.ylabel("posição (px)")
    plt.title("Ajuste em pixels com envoltória")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outdir/"x_vs_t.png"); plt.close()

def plot_dx_vs_t_metros(df: pd.DataFrame, fit: FitResult, px2m: float, outdir: Path, tmax: float = None):
    if not (px2m and px2m > 0): return

    # Conversão de pixels para metros (horizontal):
    #   Δx_m(t) = ( x(t) - C ) * px2m
    t = df["t"].values
    x = df["x"].values
    y_hat = model_damped_cos(t, fit.A, fit.b, fit.omega, fit.phi, fit.C)

    if tmax:
        mask = t <= tmax
        t, x, y_hat = t[mask], x[mask], y_hat[mask]

    dx_m = (x - fit.C) * px2m
    dx_fit_m = (y_hat - fit.C) * px2m

    # Envelopes em metros:
    #   ± (A e^{-b t}) * px2m
    env_m = (fit.A*np.exp(-fit.b*t)) * px2m

    plt.figure(figsize=(8.4, 4.4))
    plt.plot(t, dx_m, '.', ms=2, label="Δx(t) dados")
    plt.plot(t, dx_fit_m, '-', lw=1.5, label="ajuste OHA")
    plt.plot(t, +env_m, '--', lw=1, color='red', label="envelope")
    plt.plot(t, -env_m, '--', lw=1, color='red')
    plt.ylabel("Δx (m)"); plt.xlabel("t (s)")
    plt.title("Oscilação horizontal Δx(t) em metros")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outdir/"delta_x_vs_t.png"); plt.close()

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Pêndulo simples — versão enxuta")
    ap.add_argument("--csv", required=True, help="tracking_xy.csv (frame,x,y,fps[,t])")
    ap.add_argument("--saida", default="./resultados", help="Pasta de saída")
    ap.add_argument("--px2m", type=float, default=None, help="m/px")
    ap.add_argument("--L", type=float, default=None, help="Comprimento do fio (m, pivô→CM)")
    ap.add_argument("--dL", type=float, default=None, help="Incerteza de L (m)")
    ap.add_argument("--m", type=float, default=None, help="Massa (kg) opcional p/ γ e k")
    ap.add_argument("--dm", type=float, default=None, help="Incerteza da massa (kg)")
    ap.add_argument("--g", type=float, default=G_DEFAULT, help="Gravidade (m/s^2)")
    ap.add_argument("--tmax", type=float, default=None, help="Limitar gráficos até t <= tmax (s)")
    args = ap.parse_args()

    outdir = Path(args.saida); outdir.mkdir(parents=True, exist_ok=True)

    # Carrega CSV do tracking; se não houver t, calcula t = frame/fps
    df = pd.read_csv(args.csv)
    if "t" not in df.columns:
        if "fps" not in df.columns: raise ValueError("CSV precisa de 'fps' ou 't'.")
        df["t"] = df["frame"] / df["fps"]

    # Ajuste do OHA
    fit = fit_damped_cosine(df["t"].values, df["x"].values)

    # JSONs/relatórios
    salvar_parametros_json(outdir, fit)
    salvar_parametros_extras(outdir, fit, args.m, args.dm)
    (outdir / "pequenos_angulos.json").write_text(
        json.dumps(pequeno_angulo(df, args.L, args.px2m), indent=2), encoding="utf-8"
    )
    salvar_incertezas(outdir, L=args.L, dL=args.dL, m=args.m, dm=args.dm, g=args.g, fit=fit, px2m=args.px2m)

    # Gráficos
    plot_xy_trajetoria(df, outdir)
    plot_x_vs_t_pixels(df, fit, outdir, tmax=args.tmax)
    plot_dx_vs_t_metros(df, fit, args.px2m, outdir, tmax=args.tmax)

    print("Concluído. Resultados em", outdir)

if __name__ == "__main__":
    main()
