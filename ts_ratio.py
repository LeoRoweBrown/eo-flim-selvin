"""
G/(G+U) lifetimes from ThunderSTORM localisations of the two registered channels.

Input: the two ThunderSTORM CSVs of a registered stack (channel 1 = U, channel 2 = G; frames 1..6 =
phases 0, 30, ..., 150 deg) and their protocol files (pixel size, ROI offset of the analysed region).

1. Link: the molecules are immobile, so per channel all localisations within --link px across
   frames form one molecule. A molecule with two localisations in the same frame is crowded: dropped.
2. Pair U and G molecules (mode of their offsets, sm_lifetime.pair); report the residual G - U shift.
3. Select robust molecules:
   complete  detected in every frame in both channels,
   steady    per-frame U + G, divided by that frame's median over complete molecules (removes the
             global excitation fluctuation) and by the molecule's mean, stays within --max-blink
             (max |ln b_k| < ln max_blink): no blinking or bleaching across frames,
   typical   mean U + G within the middle quartiles of the complete molecules (drops dim ones and
             aggregates).
4. r_k = G_k / (G_k + U_k), shot-noise sigma_r^2 = r(1-r)/(G_k + U_k); tau = weighted fit of r_k to
   the LUT curves of the molecule's calibration pixel (lifetime_lut.py), plus the single-frame
   inversions. Ensemble: photons of the selected molecules summed per frame, fitted to their mean LUT.

Outputs (in --out): <name>_ts_molecules.csv, <name>_ts.png

Usage
-----
    conda run -n image python ts_ratio.py
    conda run -n image python ts_ratio.py <U.csv> <G.csv> --lut <..._lut.npz> --out <dir>
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import lifetime_lut as ll
import sm_lifetime as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TS = Path(r"E:\20261019\20261019_analysis\thunderSTORM")
DEFAULT_U = TS / "80mw_hilo_repeat_channel1.csv"
DEFAULT_G = TS / "80mw_hilo_repeat_channel2.csv"


def read_ts(csv):
    """Localisations with y, x in full-image px (protocol pixel size and ROI offset applied), frame from 0."""
    proto = csv.with_name(csv.stem + "-protocol.txt").read_text()
    px = json.loads(re.search(r"Camera settings:\s*(\{.*?\})", proto, re.S).group(1))["pixelSize"]
    roi = re.search(r'"roiBounds":\s*\{\s*"x":\s*(\d+),\s*"y":\s*(\d+)', proto)
    oy, ox = (int(roi.group(2)), int(roi.group(1))) if roi else (0, 0)
    d = pd.read_csv(csv)
    return pd.DataFrame(dict(frame=d["frame"].astype(int) - 1, y=d["y [nm]"] / px + oy, x=d["x [nm]"] / px + ox,
                             N=d["intensity [photon]"], sigma=d["sigma [nm]"] / px, bg=d["offset [photon]"]))


def link(loc, n_frames, radius):
    """Molecules from localisations within radius across frames: positions (m, 2), photons (m, n_frames)."""
    p = loc[["y", "x"]].to_numpy()
    pairs = cKDTree(p).query_pairs(radius, output_type="ndarray")
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(p),) * 2)
    _, lab = connected_components(adj, directed=False)
    loc = loc.assign(mol=lab)
    crowded = loc.groupby(["mol", "frame"]).size().groupby("mol").max() > 1
    loc = loc[~loc.mol.isin(crowded.index[crowded])]
    pos = loc.groupby("mol")[["y", "x"]].mean()
    N = loc.pivot(index="mol", columns="frame", values="N").reindex(index=pos.index, columns=range(n_frames))
    return pos.to_numpy(), N.to_numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("u_csv", type=Path, nargs="?", default=DEFAULT_U, help="ThunderSTORM CSV of channel 1 (U)")
    p.add_argument("g_csv", type=Path, nargs="?", default=DEFAULT_G, help="ThunderSTORM CSV of channel 2 (G)")
    p.add_argument("--lut", type=Path, default=sm.DEFAULT_LUT, help="<cal>_lut.npz from lifetime_lut.py")
    p.add_argument("--out", type=Path, default=sm.DEFAULT_OUT)
    p.add_argument("--n-frames", type=int, default=6)
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    p.add_argument("--link", type=float, default=1.0, help="linking radius across frames [px]")
    p.add_argument("--search", type=float, default=20.0, help="max G - U offset searched for pairing [px]")
    p.add_argument("--max-blink", type=float, default=1.5,
                   help="steady: normalised per-frame U + G within this factor of the molecule's mean")
    p.add_argument("--quartiles", type=float, nargs=2, default=(25, 75), help="brightness percentile window")
    p.add_argument("--tau-max", type=float, default=10.0)
    p.add_argument("--dtau", type=float, default=0.02)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.u_csv.stem.removesuffix("_channel1")
    phases = np.arange(args.n_frames) * args.phase_step
    taus = np.arange(0, args.tau_max + args.dtau / 2, args.dtau)
    L = sm.load_lut(args.lut)

    lu, lg = read_ts(args.u_csv), read_ts(args.g_csv)
    pu, NU = link(lu, args.n_frames, args.link)
    pg, NG = link(lg, args.n_frames, args.link)
    i, j, shift0 = sm.pair(pu, pg, args.search)
    off = pg[j] - pu[i]
    shift = np.median(off[np.hypot(*(off - shift0).T) < 2], 0)
    print(f"{name}: localisations U {len(lu)}, G {len(lg)}; linked molecules U {len(pu)}, G {len(pg)}, paired {len(i)}; "
          f"G - U shift dy {shift[0]:+.2f} dx {shift[1]:+.2f} px")
    NU, NG, pos = NU[i], NG[j], pu[i]

    # selection
    S = NU + NG
    complete = np.isfinite(S).all(1)
    exc = np.nanmedian(S[complete], 0)
    exc /= exc.mean()                                                  # global excitation per frame
    with np.errstate(divide="ignore", invalid="ignore"):
        b = S / exc
        b /= np.nanmean(b, 1, keepdims=True)
        blink = np.nanmax(np.abs(np.log(b)), 1)
    steady = complete & (blink < np.log(args.max_blink))
    mean_S = np.nanmean(S, 1)
    lo, hi = np.percentile(mean_S[complete], args.quartiles)
    typical = (mean_S >= lo) & (mean_S <= hi)
    Rs, near = sm.lut_curves(L, pos[:, 0], pos[:, 1], phases, taus)
    sel = steady & typical & near
    print(f"  excitation per frame (median U+G of complete molecules) {np.round(exc, 2)}")
    print(f"  complete {complete.sum()}, steady (within x{args.max_blink:g}) {steady.sum()}, + brightness "
          f"{args.quartiles[0]:g}-{args.quartiles[1]:g} pct ({lo:.0f}-{hi:.0f} photons/frame) -> selected {sel.sum()}")

    # ratios and lifetimes
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (NG / S).T                                                 # (n_frame, n_mol)
        sr = np.sqrt(r * (1 - r) / S.T)
    tau, chi2 = ll.joint_fit(r, Rs, taus, 1 / sr ** 2)
    tau_f = np.stack([ll.invert(r[k], Rs[k], taus) for k in range(len(phases))])
    r_ens = NG[sel].sum(0) / S[sel].sum(0)
    R_ens = Rs[:, :, sel].mean(2, keepdims=True)
    tau_ens, chi2_ens = ll.joint_fit(r_ens[:, None], R_ens, taus, (S[sel].sum(0) / (r_ens * (1 - r_ens)))[:, None])
    for lab, m in (("complete", complete & near), ("selected", sel)):
        t = tau[m]
        print(f"  {lab:8s}: tau median {np.nanmedian(t):.2f} ns, IQR {np.nanpercentile(t, 25):.2f}-"
              f"{np.nanpercentile(t, 75):.2f} ns, fitted {np.isfinite(t).mean():.0%} of {m.sum()}, "
              f"median chi2/dof {np.nanmedian(chi2[m]):.2f}")
    print(f"  ensemble of selected: G/(G+U) {np.round(r_ens, 3)} -> tau {tau_ens[0]:.2f} ns (chi2/dof {chi2_ens[0]:.1f})")
    print("  single-frame median tau (selected): " + "  ".join(f"{ph:.0f}={np.nanmedian(tau_f[k][sel]):.2f}"
                                                         for k, ph in enumerate(phases)))

    df = pd.DataFrame(dict(y=pos[:, 0], x=pos[:, 1], complete=complete, blink=blink, mean_photons=mean_S,
                           steady=steady, typical=typical, selected=sel, tau=tau, chi2=chi2))
    for k, ph in enumerate(phases):
        df[f"NU_{ph:g}"], df[f"NG_{ph:g}"], df[f"r_{ph:g}"], df[f"tau_{ph:g}"] = NU[:, k], NG[:, k], r[k], tau_f[k]
    df.to_csv(args.out / f"{name}_ts_molecules.csv", index=False)
    plot(name, args, L, taus, phases, pos, complete, sel, blink, mean_S, lo, hi, r, tau, tau_f,
         r_ens, R_ens, tau_ens)
    print(f"wrote {args.out}")


def plot(name, args, L, taus, phases, pos, complete, sel, blink, mean_S, lo, hi, r, tau, tau_f,
         r_ens, R_ens, tau_ens):
    to_ns = 1e3 / L["f_gate"] / L["period"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    a = ax[0, 0]
    a.plot(pos[~complete, 1], pos[~complete, 0], ".", c="0.75", ms=3, label=f"incomplete ({(~complete).sum()})")
    a.plot(pos[complete & ~sel, 1], pos[complete & ~sel, 0], ".", c="C1", ms=4,
           label=f"complete, not selected ({(complete & ~sel).sum()})")
    a.plot(pos[sel, 1], pos[sel, 0], "o", c="C2", ms=4, label=f"selected ({sel.sum()})")
    a.set_ylim(a.get_ylim()[::-1])
    a.set_aspect("equal")
    a.set_xlabel("x [px]")
    a.set_ylabel("y [px]")
    a.set_title("paired molecules")
    a.legend(fontsize=8)

    a = ax[0, 1]
    a.scatter(mean_S[complete], np.exp(blink[complete]), s=6, c=np.where(sel[complete], "C2", "C1"))
    a.axhline(args.max_blink, c="k", ls="--", lw=1)
    a.axvline(lo, c="k", ls=":", lw=1)
    a.axvline(hi, c="k", ls=":", lw=1)
    a.set_xscale("log")
    a.set_xlabel("mean U + G per frame [photons]")
    a.set_ylabel("max per-frame deviation factor (excitation-normalised)")
    a.set_title("selection of complete molecules (green: selected)")

    a = ax[1, 0]
    bins = np.linspace(0, args.tau_max, 51)
    for m, c, lab in ((complete, "0.5", "complete"), (sel, "k", "selected")):
        t = tau[m]
        a.hist(t[np.isfinite(t)], bins, histtype="step", color=c, lw=2,
               label=f"{lab}: median {np.nanmedian(t):.2f} ns ({np.isfinite(t).mean():.0%} of {m.sum()} fitted)")
    for k, ph in enumerate(phases):
        tk = tau_f[k][sel]
        a.hist(tk[np.isfinite(tk)], bins, histtype="step", color=f"C{k}", lw=0.8, label=f"selected, {ph:.0f} deg only")
    a.axvline(tau_ens[0], c="r", lw=1.5, label=f"ensemble {tau_ens[0]:.2f} ns")
    a.set_xlabel("tau [ns]")
    a.set_ylabel("molecules")
    a.set_title("lifetime from G/(G+U)")
    a.legend(fontsize=7)

    a = ax[1, 1]
    q = np.nanpercentile(r[:, sel], [25, 50, 75], axis=1)
    a.fill_between(phases * to_ns, q[0], q[2], color="C2", alpha=0.3, label="selected: IQR")
    a.plot(phases * to_ns, q[1], "o-", c="C2", label="selected: median")
    a.plot(phases * to_ns, r_ens, "ks", label="ensemble (photon sum)")
    for t, ls in ((0.0, ":"), (3.5, "--"), (tau_ens[0], "-")):
        if np.isfinite(t):
            a.plot(phases * to_ns, R_ens[:, np.argmin(abs(taus - t)), 0], "k", ls=ls, lw=1, label=f"LUT tau = {t:.2f} ns")
    a.set_xlabel("delay [ns]")
    a.set_ylabel("G / (G + U)")
    a.set_title("ratio vs phase, selected molecules")
    a.legend(fontsize=8)
    for x in ax.flat:
        x.grid(alpha=0.3)
    fig.suptitle(f"{name} (ThunderSTORM photons; LUT {L['path'].stem})", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / f"{name}_ts.png", dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    main()
