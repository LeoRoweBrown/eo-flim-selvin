"""
Bulk phasor analysis of a day's registered phase scans: IRF consistency, sample lifetimes, background, power.

Inputs are <name>_registered.tif stacks (T, C, Y, X) from register_views: methylene-blue IRFs, samples
(e.g. bulk Cy5) and blanks. Power [mW], exposure [ms], TIRF position, tag and time are parsed from names
like 17,03W_2,7Vrms_cy51uM_cleanobj_4645tirf_10mW_10ms_20261030_171529.

Per stack
---------
1. Camera offset from the image margins (as modulation.py), binning (--bin).
2. r_k = G_k / (G_k + U_k) per frame k (removes excitation drift and bleaching between frames), first
   harmonic phasor h(r) = c1 / c0 per pixel and for the ROI sums (pixels calibrated in every stack).
3. Relative to an IRF at the same pixels: h = h(r) / h(r_irf), then
       tau_phase = -sign tan(arg h) / omega,  tau_mod = sqrt(1 / |h|^2 - 1) / omega,
   corrected for the reference lifetime (--tau-ref) by h -> h / H(tau_ref), H = 1 / (1 + sign i omega tau).
   The unmodulated fraction that puts h back on the semicircle along the line to the origin,
       f_dc = 1 - |h| / cos(arg h),
   is the share of the light that must be unmodulated (or >> 1/omega long-lived) at this phase.
4. IRF matching (--match power, default; the IRF changes with laser power): same power (else the nearest IRF
   power, which is then listed in the irf_power column), preferring the same TIRF group (first digit of the TIRF position), then nearest in time; --match time: same TIRF group, nearest
   in time. --fixed-irf instead references
   everything to one IRF. Each IRF is checked against the nearest same-power IRF of another series.
5. Desync check: an IRF whose raw phase is off its power-matched neighbours by more than --desync-deg
   (one frame = 60 deg of gating phase) is reported and excluded (phase stepping out of step with the camera).

Spots (beads): --spots subtracts a local median background (--spot-bg binned px, per frame and view) and keeps
the pixels whose background-subtracted U+G is above --spot-k robust sigma; the local background itself is
analysed as a second sample, <name>_localbg, so spots and the light around them can be compared.

Sparse samples: --sample-thr keeps only pixels whose U+G exceeds that fraction of the stack's 99th
percentile, for the ROI sums (sample and IRF over the same pixels) and the per-pixel statistics.

Outputs (in --out)
------------------
acquisitions/<name>.png  one standard figure per sample: intensity, tau_phase / tau_mod / f_dc maps, pixel (or spot)
                         phasors with the ROI phasor, lifetime histograms; <name>_spots.csv per spot with --spots
irf_matrix.png      every IRF referenced to every other IRF (apparent tau_phase, |h|): should be 0 and 1
irf_drift.png       raw IRF phase / modulation of r (no reference) vs time and power, hysteresis order
phasor.png          semicircle: ROI phasors of all stacks, per-pixel clouds of the samples
power.png           sample tau_phase, tau_mod, f_dc vs power, exposure, signal and time
pixels.png          per-pixel |h| and tau_phase vs signal within each sample
background.png      blank signal per mW ms against the samples, and blank-subtracted sample phasors
bulk_phasor.csv     one row per stack (ROI and pixel-median values against its matched IRF)
irf_matrix.csv      the IRF x IRF table

20 MHz excitation (50 % duty) with 40 MHz-capable gating: the square wave has only odd harmonics of 20 MHz
and an unbiased gate only even ones, so the signal comes from the gate's 20 MHz component (static retardance
bias, ~ J1(Gamma) sin(delta)). The trace then repeats every 360 deg of RF phase and 6 x 30 deg steps cover
150 deg of it: use --f-gate-mhz 20 --period 360 --harmonics 1 (a 3rd-harmonic term is ~10 % and not
separable from 6 points; it shifts tau_phase by up to ~0.5 ns, so compare 20 MHz results with that in mind).

Usage
-----
    python bulk_phasor.py --irf <reg dir> --sample <reg dir> --blank <reg dir> --out <dir>
    python bulk_phasor.py --irf <20 MHz IRF dir> --sample <20 MHz dir> --f-gate-mhz 20 --period 360 --harmonics 1 \
        --out <dir>
"""
import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import lifetime_lut as ll
import modulation as md

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAME = re.compile(r"_(\d+)tirf_(?:(.*)_)?([\d.]+)mW_([\d.]+?)ms_(\d{8})_(\d{6})")


def parse(name):
    m = NAME.search(name)
    if not m:
        raise ValueError(f"cannot parse {name}")
    hh, mm, ss = int(m[6][:2]), int(m[6][2:4]), int(m[6][4:])
    pre = name[:m.start()].split("_")
    return dict(tirf=int(m[1]), tag=(m[2] or "") + ("_high" if name.endswith("_high") else ""),
                power=float(m[3]), exposure=float(m[4]), clock=m[6], minute=hh * 60 + mm + ss / 60,
                sample=pre[2] if len(pre) > 2 else "")


def short(r):
    return f"{r['clock']} {r['power']:g}mW {r['exposure']:g}ms {r['tirf']} {r['tag']}".strip()


class Phasor:
    def __init__(self, args):
        self.sign, self.omega = args.phase_sign, 2 * np.pi * args.f_gate_mhz * 1e-3
        self.phases = np.arange(6) * args.phase_step
        self.P = np.linalg.pinv(md.design(self.phases, args.period, args.harmonics))
        self.ref = ll.decay(np.array([1]), args.tau_ref, dict(sign=self.sign, omega=self.omega))[0, 0]

    def h1(self, r):
        c = np.tensordot(self.P, r, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (c[1] - 1j * c[2]) / c[0]

    def rel(self, h, h_irf):
        """Sample phasor relative to the IRF, including the reference dye's own decay."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return h / h_irf * self.ref

    def taus(self, h):
        with np.errstate(invalid="ignore", divide="ignore"):
            ph = -self.sign * np.tan(np.angle(h)) / self.omega
            mo = np.sqrt(np.clip(1 / np.abs(h) ** 2 - 1, 0, None)) / self.omega
            fdc = 1 - np.abs(h) / np.cos(np.angle(h))
        return ph, mo, fdc

    def gs(self, h):
        """Semicircle coordinates (g, s) with s > 0 for a delay."""
        return h.real, -self.sign * h.imag


def load(path, args):
    reg = tifffile.imread(path).astype(np.float32)
    U, G = reg[:, 0], reg[:, 1]
    off = md.background(U, args.bg_margin)
    U, G = U - off[:, None, None], G - off[:, None, None]
    return md.bin_stack(U, args.bin), md.bin_stack(G, args.bin), float(off.mean())


def local_bg(X, size):
    """Per-frame median filter (NaN-safe: uncovered pixels filled with the frame median, then restored)."""
    out = np.empty_like(X)
    for k, x in enumerate(X):
        bad = ~np.isfinite(x)
        out[k] = ndi.median_filter(np.where(bad, np.nanmedian(x), x), size)
        out[k][bad] = np.nan
    return out


def ratio(U, G):
    with np.errstate(invalid="ignore", divide="ignore"):
        return G / (U + G)


# ---------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--irf", type=Path, nargs="+", required=True, help="registered IRF stacks or folders")
    p.add_argument("--sample", type=Path, nargs="+", required=True, help="registered sample stacks or folders")
    p.add_argument("--blank", type=Path, nargs="*", default=[], help="registered blank stacks or folders")
    p.add_argument("--exclude", nargs="*", default=[], help="skip stacks whose name contains any of these")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--match", choices=("time", "power"), default="power",
                   help="IRF per sample: nearest in time in the same TIRF group, or same power first")
    p.add_argument("--fixed-irf", default=None, help="reference every stack to the IRF whose name contains this")
    p.add_argument("--desync-deg", type=float, default=30.0, help="IRF raw phase outlier threshold [deg]")
    p.add_argument("--f-gate-mhz", type=float, default=40.0)
    p.add_argument("--period", type=float, default=180.0)
    p.add_argument("--phase-step", type=float, default=30.0)
    p.add_argument("--phase-sign", type=int, default=-1, choices=(1, -1))
    p.add_argument("--harmonics", type=int, default=2,
                   help="harmonics in the per-pixel fit (20 MHz excitation with 6 x 30 deg steps: 1, period 360)")
    p.add_argument("--tau-ref", type=float, default=0.0, help="reference (methylene blue) lifetime [ns]")
    p.add_argument("--bin", type=int, default=8)
    p.add_argument("--mask-thr", type=float, default=0.3)
    p.add_argument("--spots", action="store_true", help="samples are spots (beads): local background subtraction")
    p.add_argument("--spot-bg", type=int, default=9, help="median filter size for the local background [binned px]")
    p.add_argument("--spot-k", type=float, default=5.0, help="spot pixels: above k x robust sigma after subtraction")
    p.add_argument("--sample-thr", type=float, default=0.0,
                   help="samples: use pixels with U+G > thr x their 99th percentile (beads: ~0.3)")
    p.add_argument("--bg-margin", type=int, default=250)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ph = Phasor(args)

    def files(paths):
        return [f for d in paths for f in (sorted(d.glob("*_registered.tif")) if d.is_dir() else [d])]

    rows, data = [], {}
    for kind, paths in (("irf", args.irf), ("sample", args.sample), ("blank", args.blank)):
        for f in files(paths):
            name = f.name.removesuffix("_registered.tif")
            if any(x in name for x in args.exclude):
                continue
            U, G, off = load(f, args)
            if kind == "sample" and args.spots:
                bU, bG = local_bg(U, args.spot_bg), local_bg(G, args.spot_bg)
                data[name + "_localbg"] = (bU, bG)
                rows.append(dict(name=name + "_localbg", kind=kind, offset=off, spot=False, **parse(name)))
                U, G = U - bU, G - bG
            data[name] = (U, G)
            rows.append(dict(name=name, kind=kind, offset=off, spot=kind == "sample" and args.spots, **parse(name)))
            print(f"loaded {kind:6s} {name}")
    df = pd.DataFrame(rows)

    # calibrated pixels: covered and bright in every IRF, eroded
    mask = np.ones(next(iter(data.values()))[0].shape[1:], bool)
    for n in df.name[df.kind == "irf"]:
        U, G = data[n]
        N = (U + G).mean(0)
        mask &= np.isfinite(N) & (N > args.mask_thr * np.nanpercentile(N, 99))
    mask = ndi.binary_erosion(mask, iterations=2)
    print(f"{mask.sum()} calibrated pixels (bin {args.bin})")
    sel = {}
    for _, r in df.iterrows():
        N = (data[r["name"]][0] + data[r["name"]][1]).mean(0)
        if r.spot:
            v = N[mask]
            thr = args.spot_k * 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v)))
        else:
            thr = args.sample_thr * np.nanpercentile(N[mask], 99) if r.kind == "sample" else -np.inf
        sel[r["name"]] = mask & (N > thr)

    def roi_h(name, px):
        U, G = data[name]
        return ph.h1(ratio(U[:, px].sum(1), G[:, px].sum(1)))

    # per stack: signal, frame totals, raw phasors of r (pixels and ROI)
    hpix, hroi = {}, {}
    for i, r in df.iterrows():
        U, G = data[r["name"]]
        u, g = U[:, mask].sum(1), G[:, mask].sum(1)
        tot = u + g
        hpix[r["name"]] = ph.h1(ratio(U, G))
        hroi[r["name"]] = ph.h1(ratio(u, g))
        df.loc[i, "signal"] = float(tot.mean() / mask.sum())                 # ADU / binned px / frame
        df.loc[i, "per_mW_ms"] = df.loc[i, "signal"] / r.power / r.exposure
        df.loc[i, "frame_drift"] = float((tot[-1] - tot[0]) / tot.mean())
        df.loc[i, "frame_ptp"] = float(np.ptp(tot) / tot.mean())
        df.loc[i, "raw_phase_deg"] = float(np.degrees(np.angle(hroi[r["name"]])))
        df.loc[i, "raw_mod"] = float(np.abs(hroi[r["name"]]))
        df.loc[i, "U_over_G"] = float(u.sum() / g.sum())

    # desynced IRFs: raw phase far from the other IRFs at the same power
    for i, r in df[df.kind == "irf"].iterrows():
        o = df[(df.kind == "irf") & (df.power == r.power) & (df.name != r["name"])]
        if len(o):
            d = (r.raw_phase_deg - np.median(o.raw_phase_deg) + 180) % 360 - 180
            if abs(d) > args.desync_deg:
                print(f"EXCLUDED IRF {r['name']}: raw phase {d:+.0f} deg off the other {r.power:g} mW IRFs "
                      f"({d / 60:+.1f} frames), phase stepping out of step?")
                df.loc[i, "kind"] = "irf_bad"
    irf = df[df.kind == "irf"].sort_values("minute").reset_index(drop=True)

    def match(r):
        if args.fixed_irf:
            return irf[irf.name.str.contains(args.fixed_irf, regex=False)].iloc[0]
        c = irf
        if args.match == "power":
            d = (c.power - r.power).abs()
            c = c[d == d.min()]
        same = c[(c.tirf // 1000) == (r.tirf // 1000)]
        c = same if len(same) else c
        return c.loc[(c.minute - r.minute).abs().idxmin()]

    hrel = {}
    for i, r in df.iterrows():
        m = match(r)
        if r.kind.startswith("irf"):                 # an IRF against the nearest same-power IRF of another series
            o = irf[(irf.power == r.power) & (irf.name != r["name"]) & (irf.tag != r.tag)]
            o = o if len(o) else irf[irf.name != r["name"]]
            m = o.loc[(o.minute - r.minute).abs().idxmin()]
        df.loc[i, ["irf", "irf_power"]] = m["name"], m.power
        px = sel[r["name"]]
        h = hrel[r["name"]] = ph.rel(roi_h(r["name"], px), roi_h(m["name"], px))
        tp, tm, fd = ph.taus(h)
        hp = ph.rel(hpix[r["name"]][px], hpix[m["name"]][px])
        pp, pm, pf = ph.taus(hp)
        df.loc[i, ["tau_phase", "tau_mod", "f_dc", "mod"]] = tp, tm, fd, abs(h)
        df.loc[i, ["px_tau_phase", "px_tau_mod", "px_f_dc"]] = np.nanmedian(pp), np.nanmedian(pm), np.nanmedian(pf)
        df.loc[i, "px_tau_phase_iqr"] = np.subtract(*np.nanpercentile(pp, [75, 25]))
        df.loc[i, "n_px"] = int(px.sum())
        # a decay delays by 0-90 deg; far outside that the frames are out of step with the phases
        rel = -args.phase_sign * np.degrees(np.angle(h))
        df.loc[i, "desync_suspect"] = bool(r.kind == "sample" and not -20 < rel < 90)
        if df.loc[i, "desync_suspect"]:
            print(f"DESYNC? {r['name']}: phase {rel:+.0f} deg relative to its IRF")
    df.to_csv(args.out / "bulk_phasor.csv", index=False)

    # IRF x IRF
    n = len(irf)
    T, Mo = np.zeros((n, n)), np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            h = hroi[irf.name[a]] / hroi[irf.name[b]]
            T[a, b], Mo[a, b] = ph.taus(h)[0], abs(h)
    pd.DataFrame(T, index=irf.name, columns=irf.name).to_csv(args.out / "irf_matrix.csv")

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
        cols = ["kind", "clock", "tirf", "tag", "power", "exposure", "signal", "per_mW_ms", "frame_drift",
                "raw_phase_deg", "raw_mod", "tau_phase", "tau_mod", "f_dc", "px_tau_phase", "px_tau_phase_iqr"]
        print(df.sort_values(["kind", "minute"])[cols].round(3).to_string(index=False))

    plot_irf(irf, T, Mo, args)
    plot_phasor(df, hpix, hrel, sel, ph, args)
    (args.out / "acquisitions").mkdir(exist_ok=True)
    for _, r in df[df.kind == "sample"].iterrows():
        plot_acquisition(r, data, hpix, hrel, sel[r["name"]], ph, args)
    plot_power(df, args)
    plot_pixels(df, data, hpix, mask, ph, args)
    if args.blank:
        plot_background(df, data, hroi, mask, ph, args)
    print(f"wrote {args.out}")


# ------------------------------------------------------------------- plots
def plot_irf(irf, T, Mo, args):
    labels = [short(r) for _, r in irf.iterrows()]
    fig, ax = plt.subplots(1, 2, figsize=(22, 10))
    for a, M, t, cm, lim in ((ax[0], T * 1e3, "apparent tau_phase of row vs column IRF [ps]", "RdBu_r", None),
                             (ax[1], Mo, "|h| of row vs column IRF", "viridis", None)):
        if t.startswith("apparent"):
            v = np.nanpercentile(abs(M), 98)
            im = a.imshow(M, cmap=cm, vmin=-v, vmax=v)
        else:
            im = a.imshow(M, cmap=cm, vmin=np.nanmin(M), vmax=np.nanmax(M))
        a.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        a.set_yticks(range(len(labels)), labels, fontsize=7)
        a.set_title(t)
        fig.colorbar(im, ax=a, fraction=0.04)
    fig.tight_layout()
    fig.savefig(args.out / "irf_matrix.png", dpi=70)
    plt.close(fig)

    to_ps = 1e6 / args.f_gate_mhz / 360                     # gating-phase degree -> ps
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    for g, d in irf.groupby(irf.tag.str.contains("downhyst") * 10000 + irf.tirf):
        lab = f"tirf {d.tirf.iloc[0]} {d.tag.iloc[0]}"
        shift = (d.raw_phase_deg - irf.raw_phase_deg.iloc[0]) * to_ps
        ax[0, 0].plot(d.minute - irf.minute.min(), shift, "o-", label=lab)
        ax[0, 1].plot(d.minute - irf.minute.min(), d.raw_mod, "o-", label=lab)
        ax[1, 0].plot(d.power, shift, "o-", label=lab)
        ax[1, 1].plot(d.power, d.raw_mod, "o-", label=lab)
    for a, x, y in ((ax[0, 0], "minutes", "IRF phase shift [ps]"), (ax[0, 1], "minutes", "IRF |h(r)|"),
                    (ax[1, 0], "power [mW]", "IRF phase shift [ps]"), (ax[1, 1], "power [mW]", "IRF |h(r)|")):
        a.set_xlabel(x)
        a.set_ylabel(y)
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"raw first-harmonic phasor of r = G/(G+U), ROI over the calibrated pixels, vs first IRF "
                 f"(1 deg gating phase = {to_ps:.0f} ps)")
    fig.tight_layout()
    fig.savefig(args.out / "irf_drift.png", dpi=75)
    plt.close(fig)


def semicircle(a, ph, ticks=(0.25, 0.5, 1, 2, 3, 5)):
    t = np.linspace(0, np.pi, 200)
    a.plot(0.5 + 0.5 * np.cos(t), 0.5 * np.sin(t), "k", lw=0.8)
    for tau in ticks:
        h = 1 / (1 + ph.sign * 1j * ph.omega * tau)
        g, s = ph.gs(h)
        a.plot(g, s, "k.", ms=4)
        a.annotate(f"{tau:g} ns", (g, s), xytext=(3, 3), textcoords="offset points", fontsize=7)


def plot_phasor(df, hpix, hrel, sel, ph, args):
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    s = df[~df.kind.str.startswith("irf")]
    a = ax[0]
    semicircle(a, ph)
    cmap = plt.get_cmap("viridis")
    for _, r in s.iterrows():
        g, ss = ph.gs(hrel[r["name"]])
        c = cmap(np.log(r.power / 2) / np.log(20)) if r.kind == "sample" else "0.5"
        a.plot(g, ss, "o" if r.kind == "sample" else "x", c=c, ms=7 if r.exposure >= 50 else 4)
        if r.kind == "sample":
            a.annotate(f"{r.clock[:4]}", (g, ss), xytext=(3, -8), textcoords="offset points", fontsize=6)
    a.plot([0, 1], [0, 0], "k", lw=0.5)
    a.set_aspect("equal")
    a.set_xlim(-0.05, 1.1)
    a.set_ylim(-0.05, 0.6)
    a.set_title("ROI phasors vs matched IRF (colour: power 2 -> 40 mW, large: >= 50 ms, x: blank)", fontsize=9)
    a.set_xlabel("g")
    a.set_ylabel("s")
    a.grid(alpha=0.3)

    a = ax[1]
    semicircle(a, ph)
    smp = s[s.kind == "sample"]
    hs = np.concatenate([ph.rel(hpix[n][sel[n]], hpix[m][sel[n]]) for n, m in zip(smp.name, smp.irf)])
    g, ss = ph.gs(hs[np.isfinite(hs)])
    a.hist2d(g, ss, bins=(np.linspace(-0.05, 1.1, 300), np.linspace(-0.05, 0.6, 170)), cmap="magma_r", cmin=1)
    a.plot([0, 1], [0, 0], "k", lw=0.5)
    a.set_aspect("equal")
    a.set_title(f"all sample pixels (bin {args.bin}) vs matched IRF", fontsize=9)
    a.set_xlabel("g")
    a.grid(alpha=0.3)
    fig.suptitle(f"first-harmonic phasor of r = G/(G+U), {args.f_gate_mhz:g} MHz gating, tau_ref {args.tau_ref:g} ns")
    fig.tight_layout()
    fig.savefig(args.out / "phasor.png", dpi=75)
    plt.close(fig)


def plot_acquisition(r, data, hpix, hrel, px, ph, args):
    """Standard per-acquisition figure: intensity, lifetime maps, phasor, histograms. Per pixel, or per spot
    (U and G summed over each connected spot, the IRF over the same pixels) for --spots; spots also go to a CSV."""
    U, G = data[r["name"]]
    N = (U + G).mean(0)
    hr = hrel[r["name"]]
    rp, rm, rf = ph.taus(hr)
    if r.spot:
        lab, n = ndi.label(px)
        idx = np.arange(1, n + 1)

        def sums(name):
            return [np.stack([ndi.sum(np.nan_to_num(X[k]), lab, idx) for k in range(len(X))]) for X in data[name]]

        (us, gs), (ui, gi) = sums(r["name"]), sums(r.irf)
        hv = ph.rel(ph.h1(ratio(us, gs)), ph.h1(ratio(ui, gi)))
        nv = (us + gs).mean(0)
        show = ndi.grey_dilation(lab, size=5)                 # spots enlarged for the maps
        h = np.full(N.shape, np.nan, complex)
        h[show > 0] = hv[show[show > 0] - 1]
        unit = "spot"
        yx = np.array(ndi.center_of_mass(np.nan_to_num(N), lab, idx)) * args.bin
        vp, vm, vf = ph.taus(hv)
        pd.DataFrame(dict(y=yx[:, 0], x=yx[:, 1], n_px=ndi.sum(np.ones_like(N), lab, idx), signal=nv,
                          tau_phase=vp, tau_mod=vm, f_dc=vf)).to_csv(
            args.out / "acquisitions" / f"{r['name']}_spots.csv", index=False)
    else:
        h = np.where(px, ph.rel(hpix[r["name"]], hpix[r.irf]), np.nan)
        hv, nv, unit = h[px], N[px], "pixel"
        vp, vm, vf = ph.taus(hv)
    tp, tm, fd = ph.taus(h)
    top = max(2.5, float(np.ceil(np.nanpercentile(vp, 90))))

    fig = plt.figure(figsize=(20, 10))
    gsp = fig.add_gridspec(2, 4)
    a = fig.add_subplot(gsp[0, 0])
    im = a.imshow(N, cmap="gray", vmin=0, vmax=np.nanpercentile(N[px], 99.5))
    a.contour(px, [0.5], colors="c", linewidths=0.5)
    fig.colorbar(im, ax=a, fraction=0.03)
    a.set_title(f"mean U+G [ADU/px/frame], analysed pixels outlined ({px.sum()})", fontsize=9)
    for k, (v, val, t, lim, cm) in enumerate(((tp, vp, "tau_phase [ns]", (0, top), "turbo"),
                                              (tm, vm, "tau_mod [ns]", (0, max(8, 2 * top)), "turbo"),
                                              (fd, vf, "unmodulated fraction f_dc", (0, 0.6), "magma"))):
        a = fig.add_subplot(gsp[0, k + 1])
        a.set_facecolor("0.85")
        im = a.imshow(v, cmap=cm, vmin=lim[0], vmax=lim[1], interpolation="nearest")
        fig.colorbar(im, ax=a, fraction=0.03)
        a.set_title(f"{t}, {unit} median {np.nanmedian(val):.2f}", fontsize=9)
    for a in fig.axes:
        a.set_xticks([])
        a.set_yticks([])

    a = fig.add_subplot(gsp[1, 0:2])
    semicircle(a, ph, ticks=(0.25, 0.5, 1, 2, 3, 5, 10))
    g, s = ph.gs(hv)
    ok = np.isfinite(g) & np.isfinite(s)
    if r.spot:
        a.scatter(g[ok], s[ok], s=4 + 60 * nv[ok] / np.nanpercentile(nv, 99), c="C3", alpha=0.4, lw=0)
    else:
        a.hist2d(g[ok], s[ok], bins=(np.linspace(-0.05, 1.15, 361), np.linspace(-0.1, 0.6, 211)), cmap="magma_r",
                 cmin=1e-9, weights=nv[ok])
    a.plot(*ph.gs(hr), "c+", ms=18, mew=2.5,
           label=f"ROI sum: tau_phase {rp:.2f} ns, tau_mod {rm:.2f} ns, f_dc {rf:.2f}")
    a.plot([0, 1], [0, 0], "k", lw=0.5)
    a.set_aspect("equal")
    a.set_xlim(-0.05, 1.15)
    a.set_ylim(-0.1, 0.6)
    a.set_xlabel("g")
    a.set_ylabel("s")
    a.legend(fontsize=9, loc="upper left")
    a.set_title(f"{unit} phasors ({ok.sum()}, {'size' if r.spot else 'weight'} ~ intensity) vs the power-matched IRF",
                fontsize=9)
    a.grid(alpha=0.3)

    a = fig.add_subplot(gsp[1, 2])
    bins = np.linspace(-0.5, max(8, 2 * top), 171 if not r.spot else 61)
    for v, lab_, c in ((vp, "tau_phase", "C0"), (vm, "tau_mod", "C1")):
        x = v[np.isfinite(v)]
        a.hist(np.clip(x, bins[0], bins[-1]), bins, histtype="step", lw=1.8, color=c,
               label=f"{lab_}: median {np.median(x):.2f}, IQR {np.subtract(*np.percentile(x, [75, 25])):.2f} ns")
    a.set_xlabel("tau [ns]")
    a.set_ylabel(f"{unit}s")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = fig.add_subplot(gsp[1, 3])
    a.plot(nv, vp, "." if r.spot else ",", c="C0", alpha=0.5, ms=3)
    a.set_ylim(-0.5, top + 1)
    if r.spot:
        a.set_xscale("log")
    a.set_xlabel(f"U+G per {unit} [ADU/frame{'' if r.spot else '/px'}]")
    a.set_ylabel("tau_phase [ns]")
    a.grid(alpha=0.3)
    a.set_title(f"tau_phase vs {unit} intensity", fontsize=9)
    flag = "   ** PHASE-STEP DESYNC SUSPECTED **" if r.get("desync_suspect") else ""
    fig.suptitle(f"{r['name']}\nIRF {r.irf}  (bin {args.bin}, {args.f_gate_mhz:g} MHz, tau_ref {args.tau_ref:g} ns)"
                 f"{flag}", fontsize=10, color="r" if flag else "k")
    fig.tight_layout()
    fig.savefig(args.out / "acquisitions" / f"{r['name']}.png", dpi=70)
    plt.close(fig)


def plot_power(df, args):
    s = df[df.kind == "sample"].copy()
    s["series"] = s.tirf.astype(str) + " " + s["sample"] + " " + s.tag.str.replace("_high", "")
    fig, ax = plt.subplots(3, 4, figsize=(22, 13))
    xs = (("power", "power [mW]", True), ("exposure", "exposure [ms]", True),
          ("signal", "U+G per binned px per frame [ADU]", True), ("minute", "clock [min]", False))
    ys = (("tau_phase", "tau_phase [ns]"), ("tau_mod", "tau_mod [ns]"), ("f_dc", "unmodulated fraction f_dc"))
    for j, (x, xl, log) in enumerate(xs):
        for i, (y, yl) in enumerate(ys):
            a = ax[i, j]
            for k, (lab, d) in enumerate(s.groupby("series")):
                a.scatter(d[x], d[y], s=(4 + 2 * np.log2(d.exposure / 5).clip(0)) ** 2, c=f"C{k}", label=lab)
            if log:
                a.set_xscale("log")
            a.set_xlabel(xl)
            a.set_ylabel(yl)
            a.grid(alpha=0.3, which="both")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"samples, ROI phasor vs matched IRF (tau_ref {args.tau_ref:g} ns); marker size ~ exposure")
    fig.tight_layout()
    fig.savefig(args.out / "power.png", dpi=70)
    plt.close(fig)


def plot_pixels(df, data, hpix, mask, ph, args):
    s = df[df.kind == "sample"].sort_values("minute").reset_index(drop=True)
    nc = 6
    nr = int(np.ceil(len(s) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4 * nc, 3.2 * nr), squeeze=False)
    for a, (_, r) in zip(ax.flat, s.iterrows()):
        U, G = data[r["name"]]
        N = (U + G).mean(0)[mask]
        h = ph.rel(hpix[r["name"]][mask], hpix[r.irf][mask])
        tp, tm, fd = ph.taus(h)
        a.plot(N, np.abs(h), ",", c="C0", alpha=0.4)
        a2 = a.twinx()
        a2.plot(N, tp, ",", c="C3", alpha=0.4)
        q = np.nanpercentile(N, np.linspace(2, 98, 9))
        idx = np.digitize(N, q)
        mid = [np.nanmedian(N[idx == k]) for k in range(1, len(q))]
        a.plot(mid, [np.nanmedian(np.abs(h)[idx == k]) for k in range(1, len(q))], "o-", c="C0", ms=3)
        a2.plot(mid, [np.nanmedian(tp[idx == k]) for k in range(1, len(q))], "o-", c="C3", ms=3)
        a.set_ylim(0.3, 1.1)
        a2.set_ylim(0, 2)
        a.set_title(short(r), fontsize=7)
        a.tick_params(labelsize=6)
        a2.tick_params(labelsize=6, colors="C3")
    for a in list(ax.flat)[len(s):]:
        a.axis("off")
    fig.suptitle("per pixel vs U+G [ADU]: |h| (blue, left) and tau_phase [ns] (red, right), lines: binned medians")
    fig.tight_layout()
    fig.savefig(args.out / "pixels.png", dpi=70)
    plt.close(fig)


def plot_background(df, data, hroi, mask, ph, args):
    """Blank signal against the samples, and samples with the power- and exposure-scaled blank removed."""
    b = df[df.kind == "blank"].sort_values("power")
    s = df[df.kind == "sample"]
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    a = ax[0]
    for t, d in b.groupby("tirf"):
        a.plot(d.power, d.signal / d.exposure, "s-", label=f"blank tirf {t}")
    for t, d in s.groupby("tirf"):
        a.plot(d.power, d.signal / d.exposure, "o", label=f"sample tirf {t}")
    a.set_xscale("log")
    a.set_yscale("symlog", linthresh=0.01)
    a.set_xlabel("power [mW]")
    a.set_ylabel("U+G per binned px per ms [ADU]")
    a.legend(fontsize=7)
    a.grid(alpha=0.3, which="both")

    a = ax[1]
    for t, d in b.groupby("tirf"):
        a.plot(d.power, d.raw_mod, "s-", label=f"blank tirf {t} |h(r)| (raw)")
    a.set_xlabel("power [mW]")
    a.set_ylabel("raw |h(r)|")
    a.set_title("blank: raw modulation of r (IRF ~ 0.5-0.6; 0 = unmodulated)", fontsize=9)
    a.legend(fontsize=7)
    a.grid(alpha=0.3)

    a = ax[2]
    semicircle(a, ph)
    rows = []
    for _, r in s.iterrows():
        cand = b[b.tirf // 1000 == r.tirf // 1000]
        if not len(cand):
            continue
        bb = cand.iloc[(cand.power - r.power).abs().argmin()]
        Ub, Gb = data[bb["name"]]
        scale = r.exposure / bb.exposure * r.power / bb.power
        U, G = data[r["name"]]
        u, g = U[:, mask].sum(1) - scale * Ub[:, mask].sum(1), G[:, mask].sum(1) - scale * Gb[:, mask].sum(1)
        h0 = ph.rel(hroi[r["name"]], hroi[r.irf])
        h1 = ph.rel(ph.h1(ratio(u, g)), hroi[r.irf])
        a.annotate("", ph.gs(h1), ph.gs(h0), arrowprops=dict(arrowstyle="->", lw=0.8))
        a.plot(*ph.gs(h0), "o", ms=3, c="C0")
        frac = float((scale * (Ub[:, mask] + Gb[:, mask]).sum(1)).mean() / (U[:, mask] + G[:, mask]).sum(1).mean())
        rows.append(dict(name=r["name"], blank=bb["name"], blank_fraction=frac,
                         tau_phase_sub=ph.taus(h1)[0], tau_mod_sub=ph.taus(h1)[1], f_dc_sub=ph.taus(h1)[2]))
    a.set_aspect("equal")
    a.set_xlim(0.4, 1.05)
    a.set_ylim(0, 0.4)
    a.set_title("samples before -> after subtracting the blank (scaled by power x exposure)", fontsize=9)
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "background.png", dpi=75)
    plt.close(fig)
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(args.out / "background_subtracted.csv", index=False)
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
