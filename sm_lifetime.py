"""
Single-molecule EO-FLIM, ratio method: localise, co-register, 2D Gaussian photometry, G/(G+U) -> tau.

Input: a registered stack from register_views (T, C, Y, X), or a raw acquisition folder / .ome.tif
that is registered in memory with --reg. Frames = RF phase steps, C0 = U (top), C1 = G (bottom
warped onto the top grid with the bead affine).

Per dataset
-----------
1. Camera offset from the image margins (as modulation.py); frame sums of U and G.
2. Detection: difference-of-Gaussians on each frame sum, maxima above --thr x robust sigma.
   U and G spots are paired via the mode of their offsets (search radius --search).
3. Photometry: per molecule a 2D Gaussian per view (A, y, x, b) on the frame sums with one sigma
   shared by U and G (--separate-sigma: one per view; an independent sigma error per view enters the
   G/U ratio as sigma^2); then per frame, shape fixed, A_k and b_k by linear least squares. Counts N_k = 2 pi sigma^2 A_k [ADU]
   with the local background removed. Crowded, uncovered or badly fitted molecules are dropped.
4. Co-registration: residual translation G - U from the fitted positions; the bead registration
   composed with it is saved as <name>_registration_corrected.json (usable with register_views --reg).
5. Lifetime: r_k = N_G / (N_G + N_U) per frame. Each frame normalises itself, so blinking between
   frames only changes how much a frame is worth: frames are weighted by 1/sigma_r^2 and frames
   with U + G below --min-frame-snr sigma (molecule off) are skipped. tau = weighted least-squares
   fit of r_k to the molecule's LUT curves R_k(tau) (lifetime_lut.py, calibration pixel at the
   molecule), plus the single-frame inversion of every frame.

Outputs (in --out): <name>_molecules.csv (one row per molecule), <name>_sm.png,
<name>_registration_corrected.json, and summary.csv over datasets.

Usage
-----
    conda run -n image python sm_lifetime.py
    conda run -n image python sm_lifetime.py <registered .tif | raw folder> [...] --lut <..._lut.npz>
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import lifetime_lut as ll
import modulation as md
import register_views as rv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_DATA = [Path(r"E:\20261019\20261019_analysis\registered\measurements"
                     r"\80mw_atto647n_300ms_eoflim_6phases_15,1mW_0,12Vpp_0,69vrms_lc_better_hilo_3746_repeat_registered.tif")]
DEFAULT_LUT = Path(r"E:\20261019\20261019_analysis\lut_results"
                   r"\pockels_15,2W_0,12Vpp_lcvr_0,69_20mW_100ms_methylene_6phase_lut.npz")
DEFAULT_OUT = Path(r"E:\20261019\20261019_analysis\sm_results")


# ------------------------------------------------------------------- loading
def load(path, args):
    """(name, U, G) offset-subtracted, (T, Y, X); raw input is registered with --reg in memory."""
    if path.name.endswith("_registered.tif"):
        reg = tifffile.imread(path).astype(np.float32)
        name = path.name.removesuffix("_registered.tif")
    else:
        file = rv.find_stack(path)
        name = path.name if path.is_dir() else file.name.removesuffix(".ome.tif")
        raw = tifffile.imread(file)
        reg = rv.register(raw.reshape(-1, *raw.shape[-2:]), *rv.load_warp(args.reg))
    if reg.ndim == 3:
        reg = reg[None]
    U, G = reg[:, 0], reg[:, 1]
    bg = md.background(U, args.bg_margin) if args.bg is None else np.full(len(U), args.bg)
    return name, U - bg[:, None, None], G - bg[:, None, None]


def load_lut(path):
    z = np.load(path, allow_pickle=False)
    _, idx = ndi.distance_transform_edt(~z["mask"], return_indices=True)    # nearest calibrated pixel
    return dict(cU=z["coef_U"], cG=z["coef_G"], mask=z["mask"], idx=idx, bin=int(z["bin"]),
                period=float(z["period"]), f_gate=float(z["f_gate_mhz"]), path=Path(path),
                model=dict(omega=2 * np.pi * float(z["f_gate_mhz"]) * 1e-3, sign=int(z["phase_sign"]),
                           tau_ref=float(z["tau_ref"])))


def lut_curves(L, y, x, phases, taus):
    """LUT curves R_k(tau) (n_frame, n_tau, n_mol) of the calibration pixel at full-res (y, x),
    and whether a calibrated pixel lies within 2 bins."""
    by = np.clip((np.asarray(y) // L["bin"]).astype(int), 0, L["mask"].shape[0] - 1)
    bx = np.clip((np.asarray(x) // L["bin"]).astype(int), 0, L["mask"].shape[1] - 1)
    ny, nx = L["idx"][0][by, bx], L["idx"][1][by, bx]
    CU, CG = ll.harmonic_coef(L["cU"][:, ny, nx]), ll.harmonic_coef(L["cG"][:, ny, nx])
    Rs = np.stack([ll.lut(CU, CG, 2 * np.pi * ph / L["period"], taus, L["model"]) for ph in phases])
    return Rs, np.hypot(ny - by, nx - bx) <= 2


# ------------------------------------------------------- detection, pairing
def dog(a):
    a = np.nan_to_num(a, nan=float(np.nanmedian(a)))
    return ndi.gaussian_filter(a, 1.5) - ndi.gaussian_filter(a, 8)


def detect(img, valid, thr):
    """(y, x) of local maxima of the DoG image above thr x robust sigma (pixel precision)."""
    d = dog(img)
    v = d[valid]
    s = 1.4826 * np.median(np.abs(v - np.median(v)))
    return np.argwhere((d == ndi.maximum_filter(d, 9)) & (d > thr * s) & valid).astype(float)


def pair(pu, pg, search):
    """Indices (i, j) of U/G spot pairs: shift = mode of all offsets within search, then nearest within 2 px."""
    tree = cKDTree(pg)
    off = np.array([pg[j] - pu[i] for i, js in enumerate(tree.query_ball_point(pu, search)) for j in js]).reshape(-1, 2)
    if len(off) == 0:
        raise ValueError("no U/G spot pairs within --search")
    H, ey, ex = np.histogram2d(off[:, 0], off[:, 1], bins=int(2 * search), range=[[-search, search]] * 2)
    iy, ix = np.unravel_index(np.argmax(H), H.shape)
    shift = np.array([(ey[iy] + ey[iy + 1]) / 2, (ex[ix] + ex[ix + 1]) / 2])
    d, j = tree.query(pu + shift, distance_upper_bound=2.0)
    i = np.flatnonzero(np.isfinite(d))
    return i, j[i], shift


def corrected_warp(bead_warp, shift):
    """Bead warp (cv2, top (x, y) -> bottom-local) composed with registered-G = U + shift (y, x)."""
    W = np.asarray(bead_warp)
    return np.c_[W[:, :2], W[:, :2] @ shift[::-1] + W[:, 2]]


# ---------------------------------------------------------------- photometry
def fit_spot(img, y, x, h):
    """2D Gaussian on the (2h+1)^2 window around (y, x): (A, y, x, sigma, b) or None."""
    iy, ix = int(round(y)), int(round(x))
    if iy - h < 0 or ix - h < 0 or iy + h + 1 > img.shape[0] or ix + h + 1 > img.shape[1]:
        return None
    w = img[iy - h:iy + h + 1, ix - h:ix + h + 1]
    if not np.isfinite(w).all():
        return None
    gy, gx = np.mgrid[iy - h:iy + h + 1, ix - h:ix + h + 1].astype(float)

    def resid(p):
        return (p[0] * np.exp(-((gy - p[1]) ** 2 + (gx - p[2]) ** 2) / (2 * p[3] ** 2)) + p[4] - w).ravel()
    b0 = np.percentile(w, 20)
    r = least_squares(resid, [w.max() - b0, y, x, 1.5, b0],
                      bounds=([0, y - 2, x - 2, 0.6, -np.inf], [np.inf, y + 2, x + 2, 4.0, np.inf]))
    return r.x


def fit_pair(imu, img, pu, pg, h, kappa=1.0):
    """Joint 2D Gaussians on the U and G frame sums with one shared width, sigma_G = kappa sigma_U (the
    views image the same PSF up to a fixed scale, and an independent per-view sigma error enters the
    counts as sigma^2): (A, y, x, sigma, b) for U and for G, or None."""
    wins = []
    for img_, (y, x) in ((imu, pu), (img, pg)):
        iy, ix = int(round(y)), int(round(x))
        if iy - h < 0 or ix - h < 0 or iy + h + 1 > img_.shape[0] or ix + h + 1 > img_.shape[1]:
            return None
        w = img_[iy - h:iy + h + 1, ix - h:ix + h + 1]
        if not np.isfinite(w).all():
            return None
        wins.append((w, *np.mgrid[iy - h:iy + h + 1, ix - h:ix + h + 1].astype(float)))

    def resid(p):
        return np.concatenate([(a * np.exp(-((gy - y) ** 2 + (gx - x) ** 2) / (2 * s ** 2)) + b - w).ravel()
                               for (w, gy, gx), (a, y, x, b), s in
                               zip(wins, (p[[0, 1, 2, 7]], p[[3, 4, 5, 8]]), (p[6], kappa * p[6]))])
    (wu, *_), (wg, *_) = wins
    bu, bg = np.percentile(wu, 20), np.percentile(wg, 20)
    (yu, xu), (yg, xg) = pu, pg
    r = least_squares(resid, [wu.max() - bu, yu, xu, wg.max() - bg, yg, xg, 1.5, bu, bg],
                      bounds=([0, yu - 2, xu - 2, 0, yg - 2, xg - 2, 0.6, -np.inf, -np.inf],
                              [np.inf, yu + 2, xu + 2, np.inf, yg + 2, xg + 2, 4.0, np.inf, np.inf]))
    p = r.x
    return p[[0, 1, 2, 6, 7]], np.r_[p[[3, 4, 5]], kappa * p[6], p[8]]


def frame_counts(stack, fit, h):
    """Per-frame integrated counts 2 pi s^2 A_k, their 1-sigma errors and the local background b_k [ADU/px],
    shape fixed from fit."""
    _, y0, x0, s, _ = fit
    iy, ix = int(round(y0)), int(round(x0))
    gy, gx = np.mgrid[iy - h:iy + h + 1, ix - h:ix + h + 1].astype(float)
    D = np.c_[np.exp(-((gy - y0) ** 2 + (gx - x0) ** 2) / (2 * s ** 2)).ravel(), np.ones(gy.size)]
    W = stack[:, iy - h:iy + h + 1, ix - h:ix + h + 1].reshape(len(stack), -1).T
    coef = np.linalg.lstsq(D, W, rcond=None)[0]
    var = ((W - D @ coef) ** 2).sum(0) / (len(D) - 2)
    k = 2 * np.pi * s ** 2
    return k * coef[0], k * np.sqrt(var * np.linalg.inv(D.T @ D)[0, 0]), coef[1]


# ---------------------------------------------------------------------- main
def process(path, args, L, taus):
    name, U, G = load(path, args)
    phases = np.arange(len(U)) * args.phase_step
    Us, Gs = U.sum(0), G.sum(0)
    valid = ndi.binary_erosion(np.isfinite(Gs), iterations=args.half + 2)
    valid[:, :args.bg_margin] = valid[:, -args.bg_margin:] = False
    pu, pg = detect(Us, valid, args.thr), detect(Gs, valid, args.thr)
    iu, jg, shift0 = pair(pu, pg, args.search)
    tree = cKDTree(np.r_[pu, pg])
    isolated = np.array([len(tree.query_ball_point(pu[i], args.min_sep)) <= 2 for i in iu])  # itself in U and G

    pairs = list(zip(iu[isolated], jg[isolated]))
    fits = [(fit_spot(Us, *pu[i], args.half), fit_spot(Gs, *pg[j], args.half)) for i, j in pairs]
    kappa = 1.0
    if not args.separate_sigma:
        # G / U width ratio of the optics, from the separate fits of well-fitted spots
        q = np.array([fg[3] / fu[3] for fu, fg in fits if fu is not None and fg is not None
                      and 0.8 < fu[3] < 3.5 and 0.8 < fg[3] < 3.5])
        kappa = float(np.median(q)) if args.sigma_ratio is None else args.sigma_ratio
        print(f"  sigma_G / sigma_U = {kappa:.3f}" + (" (median of separate fits)" if args.sigma_ratio is None else ""))
        fits = [fit_pair(Us, Gs, pu[i], pg[j], args.half, kappa) or (None, None) for i, j in pairs]

    rows, N, E, Bk = [], [], [], []
    for fu, fg in fits:
        if fu is None or fg is None:
            continue
        (nu, eu, bu), (ng, eg, bgk) = frame_counts(U, fu, args.half), frame_counts(G, fg, args.half)
        rows.append(dict(y=fu[1], x=fu[2], yG=fg[1], xG=fg[2], sigma_U=fu[3], sigma_G=fg[3]))
        N.append((nu, ng)), E.append((eu, eg)), Bk.append((bu, bgk))
    df = pd.DataFrame(rows)
    N, E = np.array(N), np.array(E)                                 # (n_mol, 2, n_frame): U, G
    NU, NG, EU, EG = N[:, 0], N[:, 1], E[:, 0], E[:, 1]
    Bk = np.array(Bk)                                               # local background per frame [ADU/px]

    # residual G - U translation from the fitted positions
    off = df[["yG", "xG"]].to_numpy() - df[["y", "x"]].to_numpy()
    use = np.hypot(*(off - shift0).T) < 2
    for _ in range(3):
        shift = np.median(off[use], 0)
        dist = np.hypot(*(off - shift).T)
        use = dist < 3 * max(np.sqrt(np.mean(dist[use] ** 2)), 0.05)
    rms = float(np.sqrt(np.mean(dist[use] ** 2)))
    if args.reg is not None:
        warp, split = rv.load_warp(args.reg)
        (args.out / f"{name}_registration_corrected.json").write_text(json.dumps(dict(
            warp=corrected_warp(warp, shift).tolist(), split=split, source=str(args.reg),
            residual_shift_yx=shift.tolist(), dataset=str(path),
            note="cv2 warpAffine matrix for WARP_INVERSE_MAP, top (x,y) -> bottom-local"), indent=2))

    # ratios per frame; frames where the molecule is off are dropped
    with np.errstate(divide="ignore", invalid="ignore"):
        S = NU + NG
        r = NG / S
        sr = np.sqrt(NG ** 2 * EU ** 2 + NU ** 2 * EG ** 2) / S ** 2
        on = S > args.min_frame_snr * np.hypot(EU, EG)
    r, sr = np.where(on, r, np.nan).T, np.where(on, sr, np.nan).T   # (n_frame, n_mol)
    tot = np.where(on, S, 0).sum(1)
    good = ((df.sigma_U > 0.8) & (df.sigma_U < 3.5) & (df.sigma_G > 0.8) & (df.sigma_G < 3.5)).to_numpy()
    good = good & (dist < 1.0) & (on.sum(1) >= args.min_frames)

    # LUT curves of each molecule's calibration pixel
    Rs, near = lut_curves(L, df.y.to_numpy(), df.x.to_numpy(), phases, taus)
    good = good & near
    tau, chi2 = ll.joint_fit(r, Rs, taus, 1 / sr ** 2)
    tau_f = np.stack([ll.invert(r[k], Rs[k], taus) for k in range(len(phases))])

    df["counts"], df["frames_on"], df["good"], df["tau"], df["chi2"] = tot, on.sum(1), good, tau, chi2
    for k, ph in enumerate(phases):
        df[f"NU_{ph:g}"], df[f"NG_{ph:g}"], df[f"r_{ph:g}"], df[f"sr_{ph:g}"] = NU[:, k], NG[:, k], r[k], sr[k]
        df[f"BU_{ph:g}"], df[f"BG_{ph:g}"] = Bk[:, 0, k], Bk[:, 1, k]
        df[f"tau_{ph:g}"] = tau_f[k]
    df.to_csv(args.out / f"{name}_molecules.csv", index=False)

    t = tau[good]
    row = dict(name=name, spots_U=len(pu), spots_G=len(pg), paired=len(iu), molecules=len(df), good=int(good.sum()),
               shift_dy=shift[0], shift_dx=shift[1], shift_rms=rms, median_counts=float(np.median(tot[good])),
               tau_median=float(np.nanmedian(t)), tau_iqr=float(np.subtract(*np.nanpercentile(t, [75, 25]))),
               tau_fitted=float(np.isfinite(t).mean()), chi2_median=float(np.nanmedian(chi2[good])))
    print(f"{name}\n  spots U {len(pu)}, G {len(pg)}, paired {len(iu)}, fitted {len(df)}, kept {good.sum()}; "
          f"G - U shift dy {shift[0]:+.2f} dx {shift[1]:+.2f} px (rms {rms:.2f})\n"
          f"  tau median {row['tau_median']:.2f} ns (IQR {row['tau_iqr']:.2f}), fitted {row['tau_fitted']:.0%} of kept, "
          f"median chi2/dof {row['chi2_median']:.2f}, median counts {row['median_counts']:.0f} ADU")
    plot(name, args, L, taus, phases, Us, df, good, off, shift, use, r, sr, Rs, tau)
    return row


def plot(name, args, L, taus, phases, Us, df, good, off, shift, use, r, sr, Rs, tau):
    to_ns = 1e3 / L["f_gate"] / L["period"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    a = ax[0, 0]
    d = dog(Us)
    a.imshow(d, cmap="gray", vmin=np.percentile(d, 1), vmax=np.percentile(d, 99.9))
    a.plot(df.x[good], df.y[good], "o", mfc="none", mec="lime", ms=7, mew=0.6, label=f"kept ({good.sum()})")
    a.plot(df.x[~good], df.y[~good], "o", mfc="none", mec="r", ms=7, mew=0.6, label=f"rejected ({(~good).sum()})")
    a.set_title("U frame sum (DoG), fitted molecules")
    a.legend(fontsize=8, loc="lower right")
    a.set_xticks([])
    a.set_yticks([])

    a = ax[0, 1]
    a.hist2d(off[use, 1], off[use, 0], bins=60, cmap="magma",
             range=[[shift[1] - 1.5, shift[1] + 1.5], [shift[0] - 1.5, shift[0] + 1.5]])
    a.plot(shift[1], shift[0], "c+", ms=14)
    a.set_xlabel("dx [px]")
    a.set_ylabel("dy [px]")
    a.set_title(f"G - U offset of fitted molecules after bead registration: dy {shift[0]:+.2f}, dx {shift[1]:+.2f} px")

    a = ax[1, 0]
    bins = np.linspace(0, args.tau_max, 51)
    t = tau[good]
    a.hist(t[np.isfinite(t)], bins, histtype="step", color="k", lw=2,
           label=f"weighted fit over frames: median {np.nanmedian(t):.2f} ns ({np.isfinite(t).mean():.0%} fitted)")
    for k, ph in enumerate(phases):
        tk = df[f"tau_{ph:g}"].to_numpy()[good]
        a.hist(tk[np.isfinite(tk)], bins, histtype="step", color=f"C{k}", lw=0.8, label=f"frame {ph:.0f} deg only")
    a.set_xlabel("tau [ns]")
    a.set_ylabel("molecules")
    a.set_title("per-molecule lifetime from G/(G+U)")
    a.legend(fontsize=7)

    a = ax[1, 1]
    order = np.argsort(-df.counts.to_numpy() * good)[:6]
    for n, k in enumerate(order):
        a.errorbar(phases * to_ns, r[:, k], sr[:, k], fmt="o", c=f"C{n}", ms=4, capsize=2,
                   label=f"{df.counts[k]:.0f} ADU: tau {tau[k]:.2f} ns")
        if np.isfinite(tau[k]):
            a.plot(phases * to_ns, Rs[:, np.argmin(abs(taus - tau[k])), k], "-", c=f"C{n}", lw=1)
        a.plot(phases * to_ns, Rs[:, 0, k], ":", c=f"C{n}", lw=0.8)
    a.set_xlabel("delay [ns]")
    a.set_ylabel("G / (G + U)")
    a.set_title("brightest kept molecules (line: LUT at fitted tau, dotted: tau = 0)")
    a.legend(fontsize=7)
    for x in (ax[0, 1], ax[1, 0], ax[1, 1]):
        x.grid(alpha=0.3)
    fig.suptitle(f"{name}  (LUT {L['path'].stem})", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / f"{name}_sm.png", dpi=70)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, nargs="*", default=DEFAULT_DATA,
                   help="registered *_registered.tif, raw acquisition folder(s) or .ome.tif")
    p.add_argument("--lut", type=Path, default=DEFAULT_LUT, help="<cal>_lut.npz from lifetime_lut.py")
    p.add_argument("--reg", type=Path, default=rv.DEFAULT_REG,
                   help="bead registration (for raw input and for the corrected json)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    p.add_argument("--thr", type=float, default=6.0, help="detection threshold [robust sigma of the DoG image]")
    p.add_argument("--search", type=float, default=20.0, help="max G - U offset searched for pairing [px]")
    p.add_argument("--half", type=int, default=5, help="fit window half-width [px]")
    p.add_argument("--separate-sigma", action="store_true",
                   help="fit the PSF width separately in U and G (default: one shared width)")
    p.add_argument("--sigma-ratio", type=float, default=None,
                   help="fixed sigma_G / sigma_U for the shared-width fit (default: median of separate fits)")
    p.add_argument("--min-sep", type=float, default=8.0, help="drop molecules with another spot closer than this [px]")
    p.add_argument("--min-frame-snr", type=float, default=3.0,
                   help="a frame counts if its U + G is above this many sigma (else: molecule off)")
    p.add_argument("--min-frames", type=int, default=3, help="keep molecules on in at least this many frames")
    p.add_argument("--tau-max", type=float, default=10.0)
    p.add_argument("--dtau", type=float, default=0.02)
    p.add_argument("--bg", type=float, default=None, help="fixed camera offset (default: from image margins)")
    p.add_argument("--bg-margin", type=int, default=250)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    L = load_lut(args.lut)
    taus = np.arange(0, args.tau_max + args.dtau / 2, args.dtau)
    rows = []
    for path in args.data:
        try:
            rows.append(process(path, args, L, taus))
        except Exception as e:  # keep going; report at the end
            print(f"{path.name}: FAILED ({e})")
    if rows:
        pd.DataFrame(rows).to_csv(args.out / "summary.csv", index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
