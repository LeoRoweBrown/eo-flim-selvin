"""
Single-molecule lifetime vs brightness, as in Bowman et al. Nat. Commun. 10, 4561 (2019), Fig. 3c: each
molecule's lifetime from the background-subtracted sum over a diffraction-limited ROI, plotted against its
photon count, with standard errors from shot noise and the measured background standard deviation.

Input: <name>_molecules.csv from sm_lifetime.py (positions, shared-width sigma, per-frame local background
BU_k, BG_k from the Gaussian fit offsets) and the registered stack <name>_registered.tif.

Per molecule and view, per frame k
-----------------------------------
1. ROI: the N_p pixels within --radius x sigma_U of the fitted position (U at (y, x), G at (yG, xG)).
   Counts S_k = (sum of the offset-subtracted ROI pixels - N_p b_k) x --e-per-adu [photons], b_k the
   Gaussian fit's local background.
2. Background standard deviation sigma_bg,k: robust standard deviation (1.4826 MAD) of the pixels in an
   annulus --annulus px beyond the ROI. It includes background shot noise, dark current and read noise as
   measured, so no camera noise model is needed.
3. Error as in the paper (Methods): var(S_k) = S_k + N_p sigma_bg,k^2, i.e. for R = G/U
   sigma_R / R = sqrt(1/G + 1/U + N_p (sigma_G^2 / G^2 + sigma_U^2 / U^2)); here for r_k = G/(G+U).
4. Aperture correction: S_k and its error divided by e, the fraction of the fitted Gaussian (sigma_G is
   wider than sigma_U) inside the ROI, so a different PSF width in the two views does not bias the ratio.
5. tau from the r phasor relative to the IRF at the molecule (phase) and from the weighted LUT fit (ratio).
   Standard error: the per-frame errors of r_k propagated linearly (numerical derivatives d tau / d r_k).
   The paper converts sigma_R through the lookup-table slope at its single gate phase; this is the same
   for six phases.
Spread over molecules: robust sigma of the ROI lifetimes against the robust sigma of the noise alone (a
mixture of N(0, SE_i^2)); intrinsic sigma = sqrt(obs^2 - noise^2).
Pixel lifetimes (CSV only): the same for each pixel within --pixel-radius x sigma_U of the peak (U and G
pixels at the same offset from their peaks, each minus b_k); 5/25/50/75/95 % per molecule.

Outputs (in --out): <name>_superpixel.csv (one row per molecule), <name>_superpixel.png (all molecules and
the --n-show brightest, numbered on the image), superpixel_summary.csv

Usage
-----
    conda run -n image python superpixel_lifetime.py <molecules.csv ...> --registered <dir> --lut <lut.npz> --out <dir>
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import norm

import lifetime_lut as ll
import modulation as md
import sm_lifetime as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TAUS = np.arange(0, 12.0001, 0.02)


def window(stack, y, x, r_out, r_in=None):
    """(stack values (n_frame, n_px) with r_in < distance <= r_out from (y, x) (r_in None: a disc), pixel offsets
    (n_px, 2) from round(y, x), distances^2) or None at the edge / outside the registered area."""
    h = int(np.ceil(r_out)) + 1
    iy, ix = int(round(y)), int(round(x))
    if iy - h < 0 or ix - h < 0 or iy + h + 1 > stack.shape[1] or ix + h + 1 > stack.shape[2]:
        return None
    dy, dx = np.mgrid[-h:h + 1, -h:h + 1]
    d2 = (iy + dy - y) ** 2 + (ix + dx - x) ** 2
    m = d2 <= r_out ** 2
    if r_in is not None:
        m &= d2 > r_in ** 2
    v = stack[:, iy - h:iy + h + 1, ix - h:ix + h + 1][:, m]
    if not np.isfinite(v).all():
        return None
    return v, np.c_[dy[m], dx[m]], d2[m]


def ratio_trace(cu, cg, vu, vg):
    with np.errstate(invalid="ignore", divide="ignore"):
        S = cu + cg
        return cg / S, np.sqrt(cg ** 2 * vu + cu ** 2 * vg) / S ** 2


def lifetimes(r, sr, Rs, R0, P, model):
    """(phase, ratio) lifetimes of ratio traces r (n_frame, n) with LUT curves Rs (n_frame, n_tau, n)."""
    c, ci = np.tensordot(P, r, 1), np.tensordot(P, R0, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = ((c[1] - 1j * c[2]) / c[0]) / ((ci[1] - 1j * ci[2]) / ci[0])
    tp = -model["sign"] * np.tan(np.angle(h)) / model["omega"]
    return tp, ll.joint_fit(r, Rs, TAUS, 1 / np.clip(sr, 1e-4, None) ** 2)[0]


def standard_errors(r, sr, Rs, R0, P, model, eps=0.01):
    """Linear propagation of the per-frame errors sr to both lifetimes: sqrt(sum_k (d tau / d r_k)^2 sr_k^2)."""
    var = [np.zeros(r.shape[1]), np.zeros(r.shape[1])]
    for k in range(len(r)):
        up, dn = r.copy(), r.copy()
        up[k] += eps
        dn[k] -= eps
        for j, (a, b) in enumerate(zip(lifetimes(up, sr, Rs, R0, P, model), lifetimes(dn, sr, Rs, R0, P, model))):
            var[j] += ((a - b) / (2 * eps) * sr[k]) ** 2
    return np.sqrt(var[0]), np.sqrt(var[1])


def robust_sd(v):
    v = v[np.isfinite(v)]
    return np.subtract(*np.percentile(v, [75, 25])) / 1.349 if len(v) > 3 else np.nan


def mixture_robust_sd(se):
    """Robust sigma (IQR / 1.349) of a mixture of N(0, se_i^2): the spread of the noise alone."""
    se = se[np.isfinite(se) & (se > 0)]
    if len(se) < 4:
        return np.nan
    x = np.linspace(-6 * np.median(se), 6 * np.median(se), 4001)
    cdf = norm.cdf(x[:, None] / se[None]).mean(1)
    return float(np.subtract(*np.interp([0.75, 0.25], cdf, x)) / 1.349)


def process(csv, args, L):
    name = csv.name.removesuffix("_molecules.csv")
    d = pd.read_csv(csv)
    d = d[d.good.astype(bool)].reset_index(drop=True)
    reg = tifffile.imread(args.registered / f"{name}_registered.tif").astype(np.float32)
    U, G = reg[:, 0], reg[:, 1]
    off = md.background(U, args.bg_margin)
    U, G = U - off[:, None, None], G - off[:, None, None]
    phases = np.arange(len(U)) * args.phase_step
    if f"BU_{phases[0]:g}" not in d:
        raise SystemExit(f"{csv}: no BU_/BG_ columns, rerun sm_lifetime.py")
    BU, BG = (d[[f"{c}_{p:g}" for p in phases]].to_numpy() for c in ("BU", "BG"))
    P = np.linalg.pinv(md.design(phases, L["period"], 2))
    Rs_all, _ = sm.lut_curves(L, d.y.to_numpy(), d.x.to_numpy(), phases, TAUS)
    g = args.e_per_adu

    rows, roi = [], dict(cu=[], cg=[], vu=[], vg=[], sbu=[], sbg=[], keep=[])
    for i, m in d.iterrows():
        rad, prad = args.radius * m.sigma_U, args.pixel_radius * m.sigma_U
        wins = [(window(U, m.y, m.x, rad), window(U, m.y, m.x, rad + args.annulus[1], rad + args.annulus[0]),
                 m.sigma_U, BU[i]),
                (window(G, m.yG, m.xG, rad), window(G, m.yG, m.xG, rad + args.annulus[1], rad + args.annulus[0]),
                 m.sigma_G, BG[i])]
        if any(w is None or a is None for w, a, _, _ in wins):
            continue
        cs = []
        for (v, _, d2), (va, _, _), s, b in wins:
            n = v.shape[1]
            sbg = 1.4826 * np.median(np.abs(va - np.median(va, 1, keepdims=True)), 1) * g      # [photons/px]
            enc = np.exp(-d2 / (2 * s ** 2)).sum() / (2 * np.pi * s ** 2)
            S = (v.sum(1) - n * b) * g
            cs.append((S / enc, (np.clip(S, 0, None) + n * sbg ** 2) / enc ** 2, sbg))
        (cu, vu, sbu), (cg, vg, sbg_) = cs
        for k, val in zip(("cu", "cg", "vu", "vg", "sbu", "sbg", "keep"), (cu, cg, vu, vg, sbu, sbg_, i)):
            roi[k].append(val)

        # pixel lifetimes: U and G pixels at the same offset from their peaks, error = shot + background sd
        pu, pg = window(U, m.y, m.x, prad), window(G, m.yG, m.xG, prad)
        common = {tuple(o): j for j, o in enumerate(pg[1])}
        ju = [j for j, o in enumerate(pu[1]) if tuple(o) in common]
        jg = [common[tuple(pu[1][j])] for j in ju]
        pxu, pxg = (pu[0][:, ju] - BU[i][:, None]) * g, (pg[0][:, jg] - BG[i][:, None]) * g
        r, sr = ratio_trace(pxu, pxg, np.clip(pxu, 0, None) + sbu[:, None] ** 2,
                            np.clip(pxg, 0, None) + sbg_[:, None] ** 2)
        k = len(ju)
        tp, tr = lifetimes(r, sr, np.repeat(Rs_all[:, :, [i]], k, 2), np.repeat(Rs_all[:, 0, [i]], k, 1), P, L["model"])
        row = dict(n_px_pixels=k)
        for lab, t in (("phase", tp), ("ratio", tr)):
            q = np.nanpercentile(t, [5, 25, 50, 75, 95]) if np.isfinite(t).sum() >= 3 else [np.nan] * 5
            for pc, v in zip((5, 25, 50, 75, 95), q):
                row[f"px_{lab}_p{pc}"] = v
        rows.append(row)

    keep = roi["keep"]
    d = d.loc[keep].reset_index(drop=True)
    cu, cg, vu, vg, sbu, sbg_ = (np.array(roi[k]).T for k in ("cu", "cg", "vu", "vg", "sbu", "sbg"))  # (n_frame, n_mol)
    Rs, R0 = Rs_all[:, :, keep], Rs_all[:, 0, keep]
    r, sr = ratio_trace(cu, cg, vu, vg)
    tp, tr = lifetimes(r, sr, Rs, R0, P, L["model"])
    se_p, se_r = standard_errors(r, sr, Rs, R0, P, L["model"])
    photons = (cu + cg).sum(0)

    out = d[["y", "x", "yG", "xG", "sigma_U", "sigma_G"]].copy()
    out["photons"], out["tau_phase"], out["tau_phase_se"] = photons, tp, se_p
    out["tau_ratio"], out["tau_ratio_se"], out["tau_gauss"] = tr, se_r, d.tau
    out = pd.concat([out, pd.DataFrame(rows).reset_index(drop=True)], axis=1)
    for j, ph in enumerate(phases):
        out[f"r_{ph:g}"], out[f"sr_{ph:g}"] = r[j], sr[j]
        out[f"bgsd_U_{ph:g}"], out[f"bgsd_G_{ph:g}"] = sbu[j], sbg_[j]
    out.to_csv(args.out / f"{name}_superpixel.csv", index=False)

    summary = []
    top = photons >= np.quantile(photons, 0.8413)
    for k in ("phase", "ratio"):
        for lab, sel in (("all", np.ones(len(out), bool)), ("top1sigma", top)):
            t, se = out[f"tau_{k}"].to_numpy()[sel], out[f"tau_{k}_se"].to_numpy()[sel]
            so, sn = robust_sd(t), mixture_robust_sd(se)
            row = dict(name=name, estimator=k, subset=lab, n=int(sel.sum()), photons_median=float(np.median(photons[sel])),
                       tau_median=float(np.nanmedian(t)), se_median=float(np.nanmedian(se)), sigma_obs=so,
                       sigma_noise=sn, sigma_intrinsic=float(np.sqrt(max(so ** 2 - sn ** 2, 0))))
            summary.append(row)
            print(f"  {k:5s} {lab:9s} n={row['n']:4d} photons {row['photons_median']:6.0f} | tau {row['tau_median']:.2f} ns, "
                  f"median SE {row['se_median']:.2f} | sigma over molecules {so:.2f}, noise {sn:.2f}, "
                  f"intrinsic {row['sigma_intrinsic']:.2f} ns")
    print(f"  background sd per pixel (median) U {np.median(sbu):.1f}, G {np.median(sbg_):.1f} photons")
    plot(name, out, U.sum(0), args)
    return summary


def plot(name, out, img, args):
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.1))
    lim = (-1, 7)
    for col, k, c in ((0, "phase", "C0"), (1, "ratio", "C2")):
        a = fig.add_subplot(gs[0, col])
        t, ok = out[f"tau_{k}"], np.isfinite(out[f"tau_{k}"])
        a.errorbar(t[ok], out.photons[ok], xerr=out[f"tau_{k}_se"][ok],
                   fmt="o", ms=2.5, c=c, ecolor=c, elinewidth=0.4, alpha=0.5, capsize=1.5, capthick=0.4)
        a.axvline(np.nanmedian(t), c="k", lw=0.8)
        a.set_yscale("log")
        a.set_xlim(*lim)
        a.set_xlabel(f"tau_{k} [ns] (ROI sum +- SE)")
        a.set_ylabel("photon count")
        a.set_title(f"all {ok.sum()} molecules, median {np.nanmedian(t):.2f} ns", fontsize=9)
        a.grid(alpha=0.3, which="both")
    a = fig.add_subplot(gs[0, 2])
    bins = np.arange(lim[0], lim[1], 0.2)
    for k, c in (("phase", "C0"), ("ratio", "C2")):
        t = out[f"tau_{k}"]
        a.hist(t[np.isfinite(t)], bins, histtype="step", lw=2, color=c, label=f"ROI tau_{k}")
    pix = out["px_phase_p50"]
    a.hist(pix[np.isfinite(pix)], bins, histtype="step", lw=1, color="C0", ls="--", label="pixel median tau_phase")
    a.set_xlabel("tau [ns]")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # Bowman-style panel: the brightest molecules, numbered, +- SE
    sel = out.sort_values("photons", ascending=False).head(args.n_show).reset_index()
    a = fig.add_subplot(gs[1, 0:2])
    a.errorbar(sel.tau_phase, sel.photons, xerr=sel.tau_phase_se, fmt="o", c="C3", ms=6, ecolor="k",
               elinewidth=0.8, capsize=4, capthick=0.8, zorder=3)
    for n, m in sel.iterrows():
        a.annotate(str(n + 1), (m.tau_phase, m.photons), xytext=(4, -10), textcoords="offset points", fontsize=7)
    a.set_yscale("log")
    a.set_xlim(*lim)
    a.set_xlabel("tau_phase [ns] (ROI sum +- SE)")
    a.set_ylabel("photon count")
    a.set_title(f"{args.n_show} brightest molecules", fontsize=9)
    a.grid(alpha=0.3, which="both")
    a = fig.add_subplot(gs[1, 2])
    a.imshow(img, cmap="gray", vmin=np.nanpercentile(img, 1), vmax=np.nanpercentile(img, 99.8))
    for n, m in sel.iterrows():
        a.add_patch(plt.Circle((m.x, m.y), args.radius * m.sigma_U * 3, fill=False, ec="C3", lw=0.6))
        a.annotate(str(n + 1), (m.x, m.y), xytext=(5, 5), textcoords="offset points", fontsize=6, color="y")
    a.set_title("U frame sum, numbered molecules (circles 3x ROI radius)", fontsize=9)
    a.axis("off")
    fig.suptitle(f"{name}: ROI radius {args.radius:g} sigma, SE from shot noise + measured background sd "
                 f"(annulus {args.annulus[0]:g}-{args.annulus[1]:g} px beyond the ROI)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / f"{name}_superpixel.png", dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, nargs="+", help="<name>_molecules.csv from sm_lifetime.py")
    p.add_argument("--registered", type=Path, required=True, help="folder with <name>_registered.tif")
    p.add_argument("--lut", type=Path, required=True, help="<cal>_lut.npz from lifetime_lut.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--radius", type=float, default=2.0, help="ROI radius for the sums [x sigma_U] (~ Airy radius)")
    p.add_argument("--annulus", type=float, nargs=2, default=(1.0, 4.0), metavar=("IN", "OUT"),
                   help="background-sd annulus, distances beyond the ROI radius [px]")
    p.add_argument("--pixel-radius", type=float, default=1.0, help="radius of the per-pixel lifetimes in the CSV [x sigma_U]")
    p.add_argument("--n-show", type=int, default=25, help="brightest molecules in the numbered panel")
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    p.add_argument("--e-per-adu", type=float, default=0.217, help="camera conversion [photoelectrons / ADU]")
    p.add_argument("--bg-margin", type=int, default=250)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    L = sm.load_lut(args.lut)
    rows = []
    for csv in args.data:
        print(csv.name.removesuffix("_molecules.csv"))
        rows += process(csv, args, L)
    pd.DataFrame(rows).to_csv(args.out / "superpixel_summary.csv", index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
