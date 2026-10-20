"""
Lifetime lookup table R(tau) = G/(G+U) from a methylene-blue phase scan (the IRF), harmonic method.

Input is a registered calibration stack from register_views (T, C, Y, X), pre-processed as in
modulation.py: camera offset from the margins, per-pixel U + G frame normalisation, binning.

Model
-----
With square-wave excitation e(t) the zero-lifetime trace is the gate correlated with e(t), not
the gate itself. A sample with decay h_tau replaces e by e * h_tau, so per harmonic n
    C_n(tau) = C_n(irf) H_n(tau) / H_n(tau_ref),   H_n = 1 / (1 + i n Omega tau),
whatever e(t) is. C_n(irf) per pixel: least-squares Fourier series in phase, DC + harmonics 1..2
(all that 6 phases support). The LUT at any phase offset follows analytically.

Choosing the phase
------------------
Figure of merit: photon-normalised lifetime precision sigma_tau sqrt(N) = sqrt(R(1-R)) / |dR/dtau|
(shot noise of the G/U split), averaged over --tau-range, for phases whose curve is monotonic from
0 to the end of that range. Reported: best offset overall and best among the measured phases.

Sign convention: --phase-sign +1 means increasing RF phase delays the gate relative to the
excitation, so longer lifetimes shift traces to larger phase.

Outputs (in --out)
------------------
<cal>_lut.png      ROI: IRF fit, LUT curves, precision vs phase offset, LUT at the best phase
<cal>_maps.png     per pixel: fit residual, R(0), precision at --tau-report, monotonic range
<cal>_lut.npz      per-pixel harmonic coefficients + settings (read by sm_lifetime.py)
<cal>_roi_lut.csv  ROI LUT table at the measured phases and the best phase

Usage
-----
    conda run -n image python lifetime_lut.py
    conda run -n image python lifetime_lut.py <cal_registered.tif> --out <dir>
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import modulation as md

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_CAL = Path(r"E:\20261019\20261019_analysis\registered\modulation_data"
                   r"\pockels_15,2W_0,12Vpp_lcvr_0,69_20mW_100ms_methylene_6phase_registered.tif")
DEFAULT_OUT = Path(r"E:\20261019\20261019_analysis\lut_results")


# ------------------------------------------------------------------ the LUT
# A trace is I(x) = Re sum_{n>=0} C_n e^{i n x}, x = 2 pi phi / period, C shape (n_harm + 1, n_px).
def harmonic_coef(coef):
    """[a0, a1, b1, a2, b2, ...] from md.fit_harmonics -> C."""
    return np.concatenate([coef[:1].astype(complex), coef[1::2] - 1j * coef[2::2]])


def decay(n, tau, model):
    """H_n(tau), shape (len(tau), len(n))."""
    return 1.0 / (1.0 + model["sign"] * 1j * np.multiply.outer(np.atleast_1d(tau), n) * model["omega"])


def traces(C, x, taus, model):
    """Intensity at gating phase x for each tau, (len(taus), n_px)."""
    n = np.arange(len(C))
    H = decay(n, taus, model) / decay(n, model["tau_ref"], model)
    return ((H * np.exp(1j * n * x)) @ C).real


def lut(CU, CG, x, taus, model):
    """R(tau) = G/(G+U) at gating phase x, (len(taus), n_px)."""
    u, g = traces(CU, x, taus, model), traces(CG, x, taus, model)
    with np.errstate(divide="ignore", invalid="ignore"):
        return g / (u + g)


def monotonic_upto(R, taus):
    """tau up to which each curve (n_tau, n_px) is strictly monotonic from tau = 0."""
    d = np.sign(np.diff(R, axis=0))
    return taus[np.cumprod((d == d[:1]) & (d != 0), axis=0).sum(0)]


def precision(R, taus):
    """Photon-normalised lifetime precision sigma_tau sqrt(N) [ns] along each curve."""
    dR = np.gradient(R, taus, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.clip(R * (1 - R), 0, None)) / np.abs(dR)


def figure_of_merit(R, taus, tau_range):
    """Mean precision over tau_range; inf where the curve is not monotonic over [0, tau_range[1]]."""
    sel = (taus >= tau_range[0]) & (taus <= tau_range[1])
    return np.where(monotonic_upto(R, taus) >= tau_range[1], precision(R, taus)[sel].mean(0), np.inf)


def invert(r, R, taus):
    """Lifetime from ratios r (n_px,) on each curve's monotonic branch from tau = 0 (else NaN)."""
    d = np.diff(R, axis=0)
    s0 = np.sign(d[0])
    ok = np.vstack([np.ones((1, R.shape[1]), bool), np.cumprod(np.sign(d) == s0, axis=0).astype(bool)])
    Rinc = np.where(ok, R * s0, np.inf)
    rinc = r * s0
    idx = (Rinc < rinc).sum(0)
    good = (idx >= 1) & (idx < ok.sum(0)) & np.isfinite(rinc) & (s0 != 0)
    i0 = np.clip(idx - 1, 0, len(taus) - 2)
    cols = np.arange(R.shape[1])
    y0, y1 = Rinc[i0, cols], Rinc[i0 + 1, cols]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = taus[i0] + (rinc - y0) / (y1 - y0) * (taus[i0 + 1] - taus[i0])
    return np.where(good, t, np.nan)


def joint_fit(r, Rs, taus, w=None):
    """tau minimising sum_k w_k (r_k - R_k(tau))^2 over frames k.

    r, w (n_frame, n_px), Rs (n_frame, n_tau, n_px); NaN frames are skipped. Returns (tau, chi2/dof);
    tau is NaN if the minimum is on the grid boundary (no single exponential fits)."""
    w = np.ones_like(r) if w is None else w
    w = np.where(np.isfinite(r) & np.isfinite(w), w, 0.0)
    chi = (w[:, None] * (Rs - np.nan_to_num(r)[:, None]) ** 2).sum(0)        # (n_tau, n_px)
    k = np.argmin(chi, 0)
    kc = np.clip(k, 1, len(taus) - 2)
    cols = np.arange(chi.shape[1])
    cm, c0, cp = chi[kc - 1, cols], chi[kc, cols], chi[kc + 1, cols]
    with np.errstate(divide="ignore", invalid="ignore"):
        dk = np.clip(np.nan_to_num(0.5 * (cm - cp) / (cm - 2 * c0 + cp)), -0.5, 0.5)
        red = c0 / np.maximum((w > 0).sum(0) - 1, 1)
    t = np.interp(kc + dk, np.arange(len(taus)), taus)
    inside = (k > 0) & (k < len(taus) - 1) & ((w > 0).sum(0) >= 2)
    return np.where(inside, t, np.nan), red


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cal", type=Path, nargs="?", default=DEFAULT_CAL, help="registered methylene-blue stack")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--f-gate-mhz", type=float, default=40.0, help="gating frequency (2x RF drive)")
    p.add_argument("--period", type=float, default=180.0, help="gating period in RF phase [deg]")
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    p.add_argument("--phase-sign", type=int, default=1, choices=(1, -1))
    p.add_argument("--tau-ref", type=float, default=0.0, help="lifetime of the reference dye [ns]")
    p.add_argument("--tau-max", type=float, default=10.0)
    p.add_argument("--dtau", type=float, default=0.02)
    p.add_argument("--tau-range", type=float, nargs=2, default=(0.5, 5.0), metavar=("LO", "HI"),
                   help="lifetime range the phase choice is optimised for [ns]")
    p.add_argument("--tau-report", type=float, default=3.5, help="lifetime for the precision map [ns]")
    p.add_argument("--bin", type=int, default=8)
    p.add_argument("--roi", type=int, default=200, help="ROI side length [full-res px]")
    p.add_argument("--roi-center", type=int, nargs=2, default=None, metavar=("Y", "X"))
    p.add_argument("--mask-thr", type=float, default=0.3, help="valid if U and G mean > thr * 99th pct")
    p.add_argument("--no-frame-norm", dest="frame_norm", action="store_false")
    p.add_argument("--bg", type=float, default=None, help="fixed camera offset (default: from image margins)")
    p.add_argument("--bg-margin", type=int, default=250)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    name = args.cal.name.removesuffix(".tif").removesuffix("_registered")
    model = dict(omega=2 * np.pi * args.f_gate_mhz * 1e-3, sign=args.phase_sign, tau_ref=args.tau_ref)
    to_ns = 1e3 / args.f_gate_mhz / args.period                 # RF deg -> ns
    taus = np.arange(0, args.tau_max + args.dtau / 2, args.dtau)

    # -------------------------------------------------------- calibration
    reg = tifffile.imread(args.cal).astype(np.float32)
    U, G = reg[:, 0], reg[:, 1]
    bg = md.background(U, args.bg_margin) if args.bg is None else np.full(len(U), args.bg)
    U, G = U - bg[:, None, None], G - bg[:, None, None]
    tot = np.ones(len(U))
    if args.frame_norm:
        U, G, tot = md.normalize_frames(U, G)
    phases = np.arange(len(U)) * args.phase_step
    nh = min(2, (len(phases) - 1) // 2)
    D = md.design(phases, args.period, nh)
    Ub, Gb = md.bin_stack(U, args.bin), md.bin_stack(G, args.bin)
    cU, cG = md.fit_harmonics(Ub, phases, args.period, nh), md.fit_harmonics(Gb, phases, args.period, nh)
    mask = np.isfinite(cU[0]) & np.isfinite(cG[0])
    mask &= (cU[0] > args.mask_thr * np.nanpercentile(cU[0], 99)) & (cG[0] > args.mask_thr * np.nanpercentile(cG[0], 99))
    mask = ndi.binary_erosion(mask, iterations=1)

    if args.roi_center is not None:
        cy, cx = args.roi_center
    else:
        cy, cx = ((np.array(ndi.center_of_mass(mask)) + 0.5) * args.bin).round().astype(int)
    h = args.roi // 2
    box = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
    u, g = U[:, box[0], box[1]].sum((1, 2)), G[:, box[0], box[1]].sum((1, 2))
    if not (np.isfinite(u).all() and np.isfinite(g).all()):
        raise SystemExit(f"ROI at (y={cy}, x={cx}) reaches outside the bottom view's footprint")
    sc = (u + g).mean()
    u, g = u / sc, g / sc
    hU, hG = np.linalg.pinv(D) @ u, np.linalg.pinv(D) @ g
    CU, CG = harmonic_coef(hU[:, None]), harmonic_coef(hG[:, None])
    print(f"calibration {name}: {len(phases)} phases, offset {bg.mean():.1f}, U+G per frame {np.round(tot, 2)}")
    print(f"  {mask.sum()} pixels (bin {args.bin}); ROI at (y={cy}, x={cx}); ROI fit rms "
          f"{np.sqrt(np.mean(np.r_[D @ hU - u, D @ hG - g] ** 2)):.4f} of mean U+G")

    # ------------------------------------------- ROI: phase offset choice
    fine = np.arange(0, args.period, 1.0)
    fom = np.array([figure_of_merit(lut(CU, CG, 2 * np.pi * ph / args.period, taus, model), taus,
                                    args.tau_range)[0] for ph in fine])
    fom_meas = np.array([fom[np.argmin(abs(fine - ph))] for ph in phases])
    ph_best = float(fine[np.argmin(fom)])
    print(f"best phase {ph_best:.0f} deg ({ph_best * to_ns:.2f} ns): sigma_tau sqrt(N) {fom.min():.2f} ns over "
          f"{args.tau_range[0]:g}-{args.tau_range[1]:g} ns; best measured phase {phases[np.argmin(fom_meas)]:.0f} deg "
          f"({fom_meas.min():.2f} ns)")
    ph_list = list(phases) + [ph_best]
    R_roi = np.stack([lut(CU, CG, 2 * np.pi * ph / args.period, taus, model)[:, 0] for ph in ph_list], 1)
    table = pd.DataFrame(R_roi, columns=[f"phase{ph:g}" for ph in ph_list])
    table.insert(0, "tau_ns", taus)
    table.to_csv(args.out / f"{name}_roi_lut.csv", index=False)
    plot_roi(name, args, taus, phases, u, g, hU, hG, fine, fom, R_roi, ph_list, to_ns)

    # ----------------------------------------------------------- per pixel
    i_rep = np.argmin(abs(taus - args.tau_report))
    xb = 2 * np.pi * ph_best / args.period
    idx = np.flatnonzero(mask)
    maps = np.full((3, mask.size), np.nan, np.float32)
    for s in range(0, len(idx), 20000):
        j = idx[s:s + 20000]
        R = lut(harmonic_coef(cU.reshape(len(cU), -1)[:, j]), harmonic_coef(cG.reshape(len(cG), -1)[:, j]),
                xb, taus, model)
        maps[:, j] = [R[0], precision(R, taus)[i_rep], monotonic_upto(R, taus)]
    maps = maps.reshape((3,) + mask.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.concatenate([np.tensordot(D, cU, 1) - Ub, np.tensordot(D, cG, 1) - Gb])
        rms = np.sqrt(np.mean(res ** 2, 0)) / (Ub + Gb).mean(0)
    print(f"per pixel at {ph_best:.0f} deg: median sigma_tau sqrt(N) at {taus[i_rep]:g} ns {np.nanmedian(maps[1]):.2f} ns, "
          f"median monotonic range {np.nanmedian(maps[2]):.2f} ns, median fit rms {np.nanmedian(rms[mask]):.4f}")
    plot_maps(name, args, mask, rms, maps, ph_best, taus[i_rep])
    np.savez_compressed(args.out / f"{name}_lut.npz", coef_U=cU.astype(np.float32), coef_G=cG.astype(np.float32),
                        mask=mask, bin=args.bin, phases=phases, period=args.period, f_gate_mhz=args.f_gate_mhz,
                        phase_sign=args.phase_sign, tau_ref=args.tau_ref, best_phase=ph_best,
                        frame_norm=args.frame_norm, calibration=str(args.cal))
    print(f"wrote {args.out}")


# ------------------------------------------------------------------- plots
def plot_roi(name, args, taus, phases, u, g, hU, hG, fine, fom, R_roi, ph_list, to_ns):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    a = ax[0, 0]
    dense = np.linspace(0, args.period, 361)
    Dd = md.design(dense, args.period, len(hU) // 2)
    for y, h, lab, col in ((u, hU, "U", "C0"), (g, hG, "G", "C3")):
        a.plot(phases * to_ns, y, "o", c=col, label=f"{lab} measured")
        a.plot(dense * to_ns, Dd @ h, "-", c=col, lw=1, label=f"{lab} harmonic fit")
    a.set_xlabel("delay [ns]")
    a.set_ylabel("ROI intensity / mean (U + G)")
    a.set_title("IRF (methylene blue), central ROI")
    a.legend(fontsize=8)

    a = ax[0, 1]
    for k, ph in enumerate(ph_list[:-1]):
        a.plot(taus, R_roi[:, k], c=f"C{k}", label=f"{ph:.0f} deg ({ph * to_ns:.1f} ns)")
    a.set_xlabel("tau [ns]")
    a.set_ylabel("G / (G + U)")
    a.set_title("LUT at the measured phases")
    a.legend(fontsize=8)

    a = ax[1, 0]
    a.plot(fine * to_ns, fom, "k")
    a.plot(phases * to_ns, [fom[np.argmin(abs(fine - ph))] for ph in phases], "ko")
    a.set_yscale("log")
    finite = fom[np.isfinite(fom)]
    if finite.size:
        a.set_ylim(finite.min() * 0.8, min(finite.max(), finite.min() * 30))
    a.set_xlabel("phase offset [ns]")
    a.set_ylabel("mean $\\sigma_\\tau\\sqrt{N}$ [ns]")
    a.set_title(f"precision over {args.tau_range[0]:g}-{args.tau_range[1]:g} ns (gaps: not monotonic)")

    a = ax[1, 1]
    a.plot(taus, R_roi[:, -1], "k", lw=2)
    a.set_xlabel("tau [ns]")
    a.set_ylabel("G / (G + U)")
    a.set_title(f"LUT at the best phase {ph_list[-1]:.0f} deg ({ph_list[-1] * to_ns:.2f} ns)")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle(f"{name}  (f_gate {args.f_gate_mhz:g} MHz, tau_ref {args.tau_ref:g} ns)")
    fig.tight_layout()
    fig.savefig(args.out / f"{name}_lut.png", dpi=75)
    plt.close(fig)


def plot_maps(name, args, mask, rms, maps, ph_best, tau_rep):
    panels = ((rms, "harmonic fit rms / mean(U+G)", "magma"),
              (maps[0], f"G/(G+U) at tau=0, {ph_best:.0f} deg", "viridis"),
              (maps[1], f"$\\sigma_\\tau\\sqrt{{N}}$ [ns] at {tau_rep:g} ns, {ph_best:.0f} deg", "viridis"),
              (maps[2], f"monotonic up to [ns], {ph_best:.0f} deg", "viridis"))
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for a, (v, t, cmap) in zip(ax.flat, panels):
        v = np.where(mask, v, np.nan)
        lo, hi = np.nanpercentile(v, [1, 99])
        im = a.imshow(v, cmap=cmap, vmin=lo, vmax=hi)
        a.set_title(f"{t} (median {np.nanmedian(v):.3g})", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.03)
    fig.suptitle(f"per-pixel LUT, {name} (bin {args.bin})")
    fig.tight_layout()
    fig.savefig(args.out / f"{name}_maps.png", dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    main()
