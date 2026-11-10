"""
Phasor plots of EO-FLIM phase scans: per pixel (bulk stacks) or per molecule (sm_lifetime.py output).

Three phasors per data set, each the n-th harmonic of a trace over the gate phase divided by the same
harmonic of the IRF trace at that pixel / molecule (so a zero-lifetime sample sits at (1, 0) and single
exponentials on the universal semicircle):
  U      U(phase) alone (offset-subtracted, not normalised)
  G      G(phase) alone
  ratio  G(phase) / (G(phase) + U(phase)): normalising each frame by U + G cancels what is common to both
         views (laser power drift, blinking, bleaching)
U and G alone carry those drifts: they move the U and G phasors in opposite directions.
Plotted as (g, s) with s > 0 for decays; the phase and modulation lifetimes of the median point are given.

With several --irf stacks, each data set is referenced to the IRF closest in time (HHMMSS timestamp in
the file names), since the gate's modulation drifts over a session; with --match-power, closest in time
among the IRFs nearest in laser power (_<n>mW_), since the IRF phase depends on power. --lut sets the
calibrated pixels and binning only.

Outputs (in --out): phasors.png (one row per data set: U, G, ratio), phasors_overview.png (median phasor
of every data set, coloured by time), phasors_summary.csv; with --maps, phasor_maps.png (bulk data: the
ratio phasor's modulation relative to the IRF, phase and modulation lifetimes over the field, and the modulation
against the local illumination); with --by-distance, phasors_by_distance.png (bulk pixels coloured by distance
from the field centre)

Usage
-----
    conda run -n image python phasor_plot.py --irf <irf_registered.tif ...> --lut <irf_lut.npz>
        --bulk <registered.tif ...> --mol <molecules.csv ...> --out <dir>
"""
import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tifffile

import lifetime_estimators as le
import modulation as md

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VIEWS = ("U", "G", "ratio")


def harmonic(M, trace, n):
    """n-th harmonic phasor c_n / c_0 of traces (n_phase, ...), n = 1 or 2."""
    c = np.tensordot(M.P, trace, 1)
    return (c[2 * n - 1] - 1j * c[2 * n]) / c[0]


def phasors(M, u, g, ui, gi, n):
    """IRF-relative phasors of U, G and G/(G+U), as plotted: decays with s > 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        hs = dict(U=harmonic(M, u, n) / harmonic(M, ui, n), G=harmonic(M, g, n) / harmonic(M, gi, n),
                  ratio=harmonic(M, g / (u + g), n) / harmonic(M, gi / (ui + gi), n))
    return {k: h.real - 1j * M.sign * h.imag for k, h in hs.items()}


def semicircle(a, M, n):
    t = np.r_[0, np.geomspace(0.05, 300, 500)]
    u = 1 / (1 - 1j * n * M.omega * t)
    a.plot(u.real, u.imag, "k-", lw=1)
    for tt in (0.5, 1, 2, 3, 4, 6, 10):
        q = 1 / (1 - 1j * n * M.omega * tt)
        a.plot(q.real, q.imag, "k.", ms=4)
        a.annotate(f"{tt:g} ns", (q.real, q.imag), xytext=(3, 3), textcoords="offset points", fontsize=7)
    a.set_aspect("equal")
    a.set_xlim(-0.1, 1.2)
    a.set_ylim(-0.25, 0.7)
    a.grid(alpha=0.3)
    a.set_xlabel("g")
    a.set_ylabel("s")


def lifetimes(h, M, n):
    """Phase and modulation lifetimes [ns] of a plotted phasor (s > 0 for decays)."""
    w = n * M.omega
    return np.tan(np.angle(h)) / w, np.sqrt(max(1 / abs(h) ** 2 - 1, 0)) / w


def timestamp(path):
    """Seconds since midnight from the last _HHMMSS in a file name, or None."""
    t = re.findall(r"_(\d{6})(?=[_.]|$)", path.name)
    return int(t[-1][:2]) * 3600 + int(t[-1][2:4]) * 60 + int(t[-1][4:]) if t else None


def power(path):
    """Laser setpoint [mW] from the last _<n>mW_ in a file name, or None."""
    p = re.findall(r"_(\d+(?:\.\d+)?)mW(?=[_.]|$)", path.name, re.IGNORECASE)
    return float(p[-1]) if p else None


def load_irf(path, M):
    reg = tifffile.imread(path).astype(np.float32)
    bg = md.background(reg[:, 0], 250)
    return tuple(md.bin_stack(reg[:, c] - bg[:, None, None], M.L["bin"]) for c in (0, 1))


def overview(rows, M, n, out):
    df = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
    t = df.t.to_numpy(dtype=float)
    norm = plt.Normalize(np.nanmin(t) / 3600, np.nanmax(t) / 3600) if np.isfinite(t).any() else None
    for a, v in zip(axs, VIEWS):
        semicircle(a, M, n)
        d = df[df.view == v]
        sc = a.scatter(d.g, d.s, c=d.t / 3600, cmap="viridis", norm=norm, s=40, edgecolors="k",
                       marker="o")
        for _, r in d.iterrows():
            a.annotate(le.short(r["name"])[-6:], (r.g, r.s), xytext=(4, 3), textcoords="offset points", fontsize=6)
        a.set_title(f"{v}: median phasor per data set", fontsize=9)
    fig.colorbar(sc, ax=axs, label="acquisition time [h]")
    fig.savefig(out / "phasors_overview.png", dpi=80)
    plt.close(fig)


def field_maps(sets, M, n, out):
    """Per bulk data set: intensity, IRF-normalised modulation |h|, phase and modulation lifetimes of the ratio
    phasor over the field, and |h| against the local illumination (the IRF's intensity at that pixel)."""
    fig, axs = plt.subplots(len(sets), 5, figsize=(30, 3.6 * len(sets)), squeeze=False, layout="constrained",
                            gridspec_kw=dict(width_ratios=(1, 1, 1, 1, 0.55)))
    w = n * M.omega
    for ax, (kind, name, hs, irf, t, ex) in zip(axs, sets):
        m = ex["mask"]
        h = hs["ratio"]
        tp = np.tan(np.angle(h)) / w
        with np.errstate(divide="ignore", invalid="ignore"):
            tm = np.sqrt(np.clip(1 / np.abs(h) ** 2 - 1, 0, None)) / w
        maps = []
        for v in (ex["I"] / np.median(ex["I"]), np.abs(h), tp, tm):
            a = np.full(m.shape, np.nan)
            a[m] = v
            maps.append(a)
        med_h, med_t, med_m = np.nanmedian(np.abs(h)), np.nanmedian(tp), np.nanmedian(tm)
        sd_m = np.subtract(*np.nanpercentile(tm, [75, 25])) / 1.349
        for a, img, cmap, lim, lab in ((ax[0], maps[0], "gray", (0, np.nanpercentile(maps[0], 99.5)), "intensity / median"),
                                       (ax[1], maps[1], "RdBu", (med_h * 0.8, med_h * 1.2), "modulation |h| (vs IRF)"),
                                       (ax[2], maps[2], "RdBu_r", (med_t - 0.5, med_t + 0.5), "tau_phase [ns]"),
                                       (ax[3], maps[3], "RdBu_r", (med_m - max(0.5, 2 * sd_m), med_m + max(0.5, 2 * sd_m)),
                                        "tau_mod [ns]")):
            im = a.imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1])
            fig.colorbar(im, ax=a, shrink=0.8, label=lab)
            a.set_xticks([])
            a.set_yticks([])
        ax[0].set_title(f"{kind}: {le.short(name)}  (IRF {le.short(irf.name.removesuffix('_registered.tif'))[-6:]})",
                        fontsize=9)
        ax[1].set_title(f"modulation: median {med_h:.3f}, robust sd {np.subtract(*np.nanpercentile(np.abs(h), [75, 25])) / 1.349 / med_h:.1%}",
                        fontsize=9)
        ax[2].set_title(f"phase lifetime: median {med_t:.2f} ns, robust sd "
                        f"{np.subtract(*np.nanpercentile(tp, [75, 25])) / 1.349:.2f} ns", fontsize=9)
        ax[3].set_title(f"modulation lifetime: median {med_m:.2f} ns, robust sd {sd_m:.2f} ns", fontsize=9)
        a = ax[4]
        ii = ex["Ii"] / np.median(ex["Ii"])
        ok = np.isfinite(h)
        a.hist2d(ii[ok], np.abs(h[ok]), bins=[np.linspace(0.3, 1.8, 60), np.linspace(med_h * 0.7, med_h * 1.3, 60)],
                 cmin=1, cmap="viridis")
        edges = np.percentile(ii[ok], np.linspace(0, 100, 11))
        b = np.digitize(ii[ok], edges[1:-1])
        a.plot([np.median(ii[ok][b == k]) for k in range(10)], [np.median(np.abs(h[ok])[b == k]) for k in range(10)],
               "o-", c="C3", ms=4)
        a.set_xlabel("local illumination (IRF intensity / median)")
        a.set_ylabel("|h|")
        a.set_title(f"corr {np.corrcoef(ii[ok], np.abs(h[ok]))[0, 1]:+.2f}", fontsize=9)
        a.grid(alpha=0.3)
    fig.suptitle(f"ratio phasor over the field (harmonic {n}): modulation relative to the IRF, phase and modulation "
                 "lifetimes", fontsize=10)
    fig.savefig(out / "phasor_maps.png", dpi=65)
    plt.close(fig)


def by_distance(sets, M, n, out, center, px_nm):
    """Per bulk data set: U, G and ratio phasors of every pixel, coloured by distance from the field centre
    (intensity-weighted centre of the IRF, or center = (y, x) [full-res px]); median phasor per distance band."""
    b = M.L["bin"]
    fig, axs = plt.subplots(len(sets), 3, figsize=(18, 5 * len(sets)), squeeze=False, layout="constrained")
    for ax, (kind, name, hs, irf, t, ex) in zip(axs, sets):
        yy, xx = np.nonzero(ex["mask"])
        yy, xx = (yy + 0.5) * b, (xx + 0.5) * b                                  # full-res px
        cy, cx = center if center is not None else (np.sum(yy * ex["Ii"]) / ex["Ii"].sum(),
                                                    np.sum(xx * ex["Ii"]) / ex["Ii"].sum())
        r = np.hypot(yy - cy, xx - cx) * px_nm / 1e3                             # um
        bands = np.percentile(r, np.linspace(0, 100, 7))
        for a, v in zip(ax, VIEWS):
            h = hs[v]
            ok = np.isfinite(h) & (np.abs(h) < 3)
            semicircle(a, M, n)
            order = np.argsort(r[ok])[::-1]                                    # centre drawn on top
            sc = a.scatter(h[ok].real[order], h[ok].imag[order], c=r[ok][order], cmap="plasma_r", s=1.5,
                           alpha=0.5, rasterized=True, vmin=0, vmax=np.percentile(r, 99))
            k = np.digitize(r, bands[1:-1])
            med = [np.median(h[ok & (k == j)].real) + 1j * np.median(h[ok & (k == j)].imag) for j in range(6)]
            a.plot([m.real for m in med], [m.imag for m in med], "k-", lw=1)
            a.scatter([m.real for m in med], [m.imag for m in med], c=[np.median(r[k == j]) for j in range(6)],
                      cmap="plasma_r", vmin=0, vmax=np.percentile(r, 99), s=60, edgecolors="k", zorder=3)
            tp = [lifetimes(m, M, n) for m in med]
            a.set_title(f"{kind}: {le.short(name)}  (IRF {le.short(irf.name.removesuffix('_registered.tif'))[-6:]})\n"
                        f"{v}: centre -> edge  tau_phase {tp[0][0]:.2f} -> {tp[-1][0]:.2f}, "
                        f"tau_mod {tp[0][1]:.2f} -> {tp[-1][1]:.2f} ns", fontsize=9)
        fig.colorbar(sc, ax=ax, label="distance from field centre [um]", shrink=0.8)
    fig.suptitle(f"harmonic {n} phasors per pixel, coloured by distance from the field centre; large dots: median "
                 "per distance sextile", fontsize=10)
    fig.savefig(out / "phasors_by_distance.png", dpi=70)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--irf", type=Path, nargs="+", required=True,
                   help="registered IRF stack(s); with several, the one closest in time is used per data set")
    p.add_argument("--lut", type=Path, required=True, help="<cal>_lut.npz of that IRF (calibrated pixels, bin)")
    p.add_argument("--bulk", type=Path, nargs="*", default=[], help="registered bulk stacks (per-pixel phasors)")
    p.add_argument("--mol", type=Path, nargs="*", default=[], help="<name>_molecules.csv from sm_lifetime.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--harmonic", type=int, default=1, choices=(1, 2))
    p.add_argument("--match-power", action="store_true",
                   help="reference each data set to the IRFs nearest in laser power (_<n>mW_ in the file names), "
                        "then nearest in time")
    p.add_argument("--maps", action="store_true",
                   help="also map the ratio phasor's modulation, phase and modulation lifetimes over the field "
                        "(bulk data)")
    p.add_argument("--by-distance", action="store_true",
                   help="also plot bulk phasors per pixel coloured by distance from the field centre")
    p.add_argument("--center", type=float, nargs=2, default=None, metavar=("Y", "X"),
                   help="field centre [full-res px] (default: intensity-weighted centre of the IRF)")
    p.add_argument("--px-nm", type=float, default=74.0, help="sample-plane pixel size [nm]")
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = args.harmonic

    M = le.Model(args.lut, args.phase_step)
    irf_t = [timestamp(f) for f in args.irf]
    if len(args.irf) > 1 and None in irf_t:
        p.error("with several --irf stacks every file name needs a _HHMMSS timestamp")
    cache = {}

    irf_p = [power(f) for f in args.irf]
    if args.match_power and None in irf_p:
        p.error("--match-power needs a _<n>mW_ laser power in every IRF file name")

    def nearest_irf(path):
        t, pw = timestamp(path), power(path)
        cand = range(len(args.irf))
        if args.match_power and pw is not None:
            dp = [abs(pw - q) for q in irf_p]
            cand = [i for i in cand if dp[i] == min(dp)]
        i = cand[0] if len(cand) == 1 or t is None else min(cand, key=lambda i: abs(t - irf_t[i]))
        if i not in cache:
            cache[i] = load_irf(args.irf[i], M)
        return args.irf[i], *cache[i]

    sets = []
    for path in args.bulk + args.mol:
        irf, Ui, Gi = nearest_irf(path)
        extra = None
        if path in args.bulk:
            u, g, ui, gi, _, m = le.bulk_sample(path, M, Ui, Gi, with_mask=True)
            kind, name = "bulk", path.name.removesuffix("_registered.tif")
            extra = dict(mask=m, I=(u + g).mean(0), Ii=(ui + gi).mean(0))
        else:
            u, g, ui, gi, _, _ = le.mol_sample(path, M, Ui, Gi)
            kind, name = "bead" if "beads" in path.name else "smlm", path.name.removesuffix("_molecules.csv")
        sets.append((kind, name, phasors(M, u, g, ui, gi, n), irf, timestamp(path), extra))

    rows = []
    fig, axs = plt.subplots(len(sets), 3, figsize=(17, 4.6 * len(sets)), squeeze=False, layout="constrained")
    for ax, (kind, name, hs, irf, t, _) in zip(axs, sets):
        for a, v in zip(ax, VIEWS):
            h = hs[v]
            h = h[np.isfinite(h) & (np.abs(h) < 3)]
            semicircle(a, M, n)
            if kind == "bulk":
                a.hist2d(h.real, h.imag, bins=[np.linspace(-0.1, 1.2, 261), np.linspace(-0.25, 0.7, 191)],
                         cmin=1, cmap="viridis")
            else:
                a.plot(h.real, h.imag, ".", c="C3" if kind == "smlm" else "C2", ms=2, alpha=0.4)
            hm = np.median(h.real) + 1j * np.median(h.imag)
            tp, tm = lifetimes(hm, M, n)
            a.plot(hm.real, hm.imag, "o", c="w", mec="k", ms=8)
            a.set_title(f"{kind}: {le.short(name)}  (IRF {le.short(irf.name.removesuffix('_registered.tif'))[-6:]})"
                        f"\n{v}: median tau_phase {tp:.2f}, tau_mod {tm:.2f} ns (n={len(h)})", fontsize=9)
            rows.append(dict(kind=kind, name=name, view=v, n=len(h), g=hm.real, s=hm.imag, tau_phase=tp, tau_mod=tm,
                             irf=irf.name, t=t))
    irfs = args.irf[0].name if len(args.irf) == 1 else f"the nearest in time of {len(args.irf)} IRFs"
    fig.suptitle(f"harmonic {n} phasors relative to {irfs}", fontsize=10)
    fig.savefig(args.out / "phasors.png", dpi=70)
    plt.close(fig)
    overview(rows, M, n, args.out)
    bulk_sets = [s for s in sets if s[5] is not None]
    if args.maps and bulk_sets:
        field_maps(bulk_sets, M, n, args.out)
    if args.by_distance and bulk_sets:
        by_distance(bulk_sets, M, n, args.out, args.center, args.px_nm)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "phasors_summary.csv", index=False)
    print(df.pivot_table(index="name", columns="view", values=["tau_phase", "tau_mod"], sort=False).round(2).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
