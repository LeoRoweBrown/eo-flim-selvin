"""
Bowman-style DFT phasor per localised bead/molecule instead of per pixel.

Same phasor as dft_bowman_method.py, but the intensities are 2D Gaussian peak heights:
1. Detect spots on the U and G frame sums (DoG, as sm_lifetime.py), pair them, and fit one 2D
   Gaussian per view (shared width, sigma_G = kappa sigma_U) on the frame sums.
2. Per frame, with the shape fixed, the peak height A_k (local background b_k removed) by linear
   least squares -> r_k = A_G / (A_G + A_U).
3. DFT of r_k over the phase sweep (dft_bowman_method.ft_data), divided by the pixelwise IRF DFT
   at the spot's pixel (IRF not peak-fitted, just pixelwise as in dft_bowman_method).

Outputs (in --out): <name>_beads.csv (one row per spot: position, sigma, peak heights, g, s,
tau_phase, tau_mod), <name>_phasor.png, and summary.csv.

Usage
-----
    python dft_bowman_method_gaussfit.py <folder or *_registered.tif ...> --irf <irf_registered.tif> --out <dir>
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from tifffile import imread

import sm_lifetime as sm
from dft_bowman_method import ft_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def localise(U, G, args):
    """Gaussian fits of paired, isolated spots: DataFrame of positions/widths, peak heights AU, AG (n, T)."""
    Us, Gs = np.nan_to_num(U.sum(0)), G.sum(0)
    valid = ndi.binary_erosion(np.isfinite(Gs), iterations=args.half + 2)
    e = args.edge
    valid[:e], valid[-e:], valid[:, :e], valid[:, -e:] = False, False, False, False
    Gs = np.nan_to_num(Gs)
    pu, pg = sm.detect(Us, valid, args.thr), sm.detect(Gs, valid, args.thr)
    iu, jg, _ = sm.pair(pu, pg, args.search)
    tree = cKDTree(np.r_[pu, pg])
    isolated = np.array([len(tree.query_ball_point(pu[i], args.min_sep)) <= 2 for i in iu], dtype=bool)
    pairs = list(zip(iu[isolated], jg[isolated]))

    sep = [(sm.fit_spot(Us, *pu[i], args.half), sm.fit_spot(Gs, *pg[j], args.half)) for i, j in pairs]
    q = [fg[3] / fu[3] for fu, fg in sep if fu is not None and fg is not None
         and 0.8 < fu[3] < 3.5 and 0.8 < fg[3] < 3.5]
    kappa = float(np.median(q)) if q else 1.0
    fits = [sm.fit_pair(Us, Gs, pu[i], pg[j], args.half, kappa) or (None, None) for i, j in pairs]

    rows, AU, AG = [], [], []
    for fu, fg in fits:
        if fu is None or fg is None:
            continue
        # frame_counts gives 2 pi s^2 A_k; divide back to the peak height A_k
        au = sm.frame_counts(U, fu, args.half)[0] / (2 * np.pi * fu[3] ** 2)
        ag = sm.frame_counts(G, fg, args.half)[0] / (2 * np.pi * fg[3] ** 2)
        rows.append(dict(y=fu[1], x=fu[2], yG=fg[1], xG=fg[2], sigma_U=fu[3], sigma_G=fg[3]))
        AU.append(au), AG.append(ag)
    df = pd.DataFrame(rows)
    AU, AG = np.array(AU), np.array(AG)
    off = (df[["yG", "xG"]] - df[["y", "x"]].to_numpy()).to_numpy()
    d = np.hypot(*(off - np.median(off, 0)).T)                     # vs the residual G - U shift
    good = ((df.sigma_U > 0.8) & (df.sigma_U < 3.5) & (df.sigma_G > 0.8) & (df.sigma_G < 3.5)).to_numpy()
    good = good & (d < 1.0) & (AU > 0).all(1) & (AG > 0).all(1)
    print(f"  spots U {len(pu)}, G {len(pg)}, paired {len(iu)}, isolated {isolated.sum()}, "
          f"fitted {len(df)}, kept {good.sum()}; sigma_G/sigma_U {kappa:.3f}")
    return df, AU, AG, good


def irf_at(dft_irf, y, x, r):
    """IRF phasor at the spot pixel (mean over a (2r+1)^2 neighbourhood if r > 0)."""
    iy = np.clip(np.round(y).astype(int), 0, dft_irf.shape[0] - 1)
    ix = np.clip(np.round(x).astype(int), 0, dft_irf.shape[1] - 1)
    if r == 0:
        return dft_irf[iy, ix]
    out = np.empty(len(iy), complex)
    for n, (a, b) in enumerate(zip(iy, ix)):
        out[n] = np.nanmean(dft_irf[max(a - r, 0):a + r + 1, max(b - r, 0):b + r + 1])
    return out


def universal_circle(ax, f_hz):
    omega = 2 * np.pi * f_hz
    tau = np.array([1e-9, 2e-9, 3e-9, 5e-9, 10e-9])
    gm = 1 / (1 + (omega * tau) ** 2)
    sm_ = gm * omega * tau
    th = np.linspace(0, np.pi, 100)
    ax.plot(0.5 + 0.5 * np.cos(th), 0.5 * np.sin(th), "k", lw=0.8)
    ax.scatter(gm, sm_, c="k", s=10)
    for g, s, t in zip(gm, sm_, tau):
        ax.text(g, s + 0.01, f"{t * 1e9:.0f} ns", fontsize=8)


def plot(name, df, f_hz, out):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    k = df[df.good]
    a = ax[0]
    sc = a.scatter(k.g, k.s, c=np.log10(k.peak_mean), s=6, cmap="viridis")
    fig.colorbar(sc, ax=a, label="log10 mean peak height (U+G) [ADU]")
    universal_circle(a, f_hz)
    a.plot(np.median(k.g), np.median(k.s), "r+", ms=14, mew=2,
           label=f"median ({len(k)} spots): tau_phase {np.nanmedian(k.tau_phase):.2f}, "
                 f"tau_mod {np.nanmedian(k.tau_mod):.2f} ns")
    a.set_xlim(-0.1, 1.2)
    a.set_ylim(-0.1, 0.7)
    a.set_aspect("equal")
    a.set_xlabel("g")
    a.set_ylabel("s")
    a.legend(fontsize=8, loc="upper right")
    a.set_title(f"{name}: per-spot phasor (Gaussian peak heights / pixelwise IRF)", fontsize=9)

    a = ax[1]
    bins = np.linspace(0, 8, 61)
    a.hist(k.tau_phase.clip(-1, 9), bins, histtype="step", lw=1.5, label="tau_phase")
    a.hist(k.tau_mod.clip(-1, 9), bins, histtype="step", lw=1.5, label="tau_mod")
    a.set_xlabel("tau [ns]")
    a.set_ylabel("spots")
    a.legend()
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{name}_phasor.png", dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, nargs="+", help="folder of *_registered.tif and/or individual files")
    p.add_argument("--pattern", default="*_registered.tif", help="glob used when data is a folder")
    p.add_argument("--irf", type=Path, required=True, help="registered IRF (methylene blue) stack")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--freq", type=float, default=20.0, help="RF frequency [MHz] of laser")
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    p.add_argument("--period", type=float, default=180.0, help="gating period in RF phase [deg]")
    p.add_argument("--bg", type=float, default=90, help="camera offset subtracted from the IRF")
    p.add_argument("--irf-threshold", type=float, default=40, help="IRF pixels below this in U or G -> NaN")
    p.add_argument("--irf-radius", type=int, default=0,
                   help="average the IRF phasor over a (2r+1)^2 neighbourhood of the spot (0: spot pixel)")
    p.add_argument("--thr", type=float, default=6.0, help="detection threshold [robust sigma of the DoG]")
    p.add_argument("--search", type=float, default=20.0, help="max G - U offset searched for pairing [px]")
    p.add_argument("--half", type=int, default=5, help="fit window half-width [px]")
    p.add_argument("--min-sep", type=float, default=8.0, help="drop spots with a neighbour closer than this [px]")
    p.add_argument("--edge", type=int, default=20, help="ignore spots this close to the image edge [px]")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = []
    for d in args.data:
        files += sorted(d.glob(args.pattern)) if d.is_dir() else [d]
    files = [f for f in files if f.resolve() != args.irf.resolve()]

    f_hz = args.freq * 1e6 * 360 / args.period     # phasor at the gate frequency
    omega = 2 * np.pi * f_hz
    irf = imread(args.irf).astype(np.float32) - args.bg
    dft_irf = ft_data(irf, args.phase_step, args.period, threshold=args.irf_threshold)

    summary = []
    for path in files:
        name = path.name.removesuffix(".tif").removesuffix("_registered")
        print(name)
        stack = imread(path).astype(np.float32)
        if stack.ndim != 4 or stack.shape[0] != irf.shape[0]:
            print(f"  skipped: shape {stack.shape} vs IRF {irf.shape}")
            continue
        try:
            df, AU, AG, good = localise(stack[:, 0], stack[:, 1], args)
        except ValueError as e:
            print(f"  FAILED ({e})")
            continue

        # (T, 2, n): same layout as a registered stack -> same DFT as the pixelwise code
        amp = np.stack([AU.T, AG.T], axis=1)
        dft = ft_data(amp, args.phase_step, args.period, threshold=-np.inf)
        ph = dft / irf_at(dft_irf, df.y.to_numpy(), df.x.to_numpy(), args.irf_radius)
        m, phi = np.abs(ph), np.angle(ph)
        df["g"], df["s"], df["mod"], df["phase_deg"] = ph.real, ph.imag, m, np.degrees(phi)
        df["tau_phase"] = np.tan(phi) / omega * 1e9
        with np.errstate(divide="ignore", invalid="ignore"):
            df["tau_mod"] = np.sqrt(1 / m ** 2 - 1) / omega * 1e9
        df["peak_mean"] = (AU + AG).mean(1)
        df["good"] = good & np.isfinite(ph)
        for k in range(AU.shape[1]):
            df[f"AU_{k}"], df[f"AG_{k}"] = AU[:, k], AG[:, k]
        df.to_csv(args.out / f"{name}_beads.csv", index=False)

        kept = df[df.good]
        tp, tm = kept.tau_phase, kept.tau_mod
        row = dict(name=name, spots=len(df), kept=len(kept), g_median=kept.g.median(), s_median=kept.s.median(),
                   tau_phase_median=tp.median(), tau_phase_iqr=tp.quantile(.75) - tp.quantile(.25),
                   tau_mod_median=tm.median(), tau_mod_iqr=tm.quantile(.75) - tm.quantile(.25),
                   peak_median=kept.peak_mean.median())
        summary.append(row)
        print(f"  g {row['g_median']:.3f} s {row['s_median']:.3f}; tau_phase {row['tau_phase_median']:.2f} ns "
              f"(IQR {row['tau_phase_iqr']:.2f}), tau_mod {row['tau_mod_median']:.2f} ns "
              f"(IQR {row['tau_mod_iqr']:.2f}), median peak {row['peak_median']:.0f} ADU")
        plot(name, df, f_hz, args.out)

    if summary:
        pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
