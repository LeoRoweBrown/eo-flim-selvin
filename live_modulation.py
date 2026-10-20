"""
Live modulation-depth maps from the microscope via pycromanager (or a saved stack, for testing).

Each cycle acquires one frame per RF phase (0, 30, ..., 150 deg by default), then as modulation.py:
split the two views at the registration's split row, warp the bottom view (G) onto the top (U)
with the registration json, subtract the camera offset (image margins), normalise every pixel so
U + G is phase-independent, bin, and fit DC + harmonics per pixel (pseudo-inverse, precomputed).
Shown: m1 maps of U, G and their mean on a fixed colour scale, the central-ROI traces with fits,
and the per-frame U + G (excitation stability). The ROI is fixed after the first cycle.

Setting the phase (Micro-Manager must be running with the pycromanager server enabled):
    --group <config group> --presets <one preset per phase>        e.g. the channel group used for
                                                                   the 6-phase acquisitions
    --device <device> --prop <property> --values <one per phase>   e.g. a function-generator phase

Usage
-----
    python live_modulation.py --group Phase --presets 0 30 60 90 120 150
    python live_modulation.py --device FG --prop Phase --values 0 30 60 90 120 150 --exposure 100
    python live_modulation.py --file <acquisition folder or .ome.tif> [--loop]   # replay, no microscope
"""
import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi

import modulation as md
import register_views as rv


# ------------------------------------------------------------------ sources
class Microscope:
    """One frame per phase from Micro-Manager, phase set by config-group presets or a device property."""

    def __init__(self, args):
        from pycromanager import Core  # only needed live
        self.core = Core()
        self.args = args
        if args.exposure is not None:
            self.core.set_exposure(args.exposure)
        self.steps = args.presets if args.group else args.values
        if not self.steps:
            raise SystemExit("give --group with --presets, or --device/--prop with --values")

    def set_phase(self, k):
        a, c = self.args, self.core
        if a.group:
            c.set_config(a.group, a.presets[k])
            c.wait_for_config(a.group, a.presets[k])
        else:
            c.set_property(a.device, a.prop, a.values[k])
            c.wait_for_device(a.device)

    def cycle(self):
        frames = []
        for k in range(len(self.steps)):
            self.set_phase(k)
            self.core.snap_image()
            im = self.core.get_tagged_image()
            frames.append(np.reshape(im.pix, (im.tags["Height"], im.tags["Width"])))
        return np.stack(frames)


class Replay:
    """Cycles of n_phases frames from a saved Micro-Manager stack."""

    def __init__(self, args):
        raw = tifffile.imread(rv.find_stack(args.file))
        self.raw = raw.reshape(-1, *raw.shape[-2:])
        self.n, self.loop, self.i = args.n_phases, args.loop, 0

    def cycle(self):
        if self.i + self.n > len(self.raw):
            if not self.loop:
                return None
            self.i = 0
        out = self.raw[self.i:self.i + self.n]
        self.i += self.n
        return out


# --------------------------------------------------------------- processing
class Processor:
    def __init__(self, args):
        self.args = args
        self.warp, self.split = rv.load_warp(args.reg)
        self.phases = np.arange(args.n_phases) * args.phase_step
        self.nh = min(args.harmonics, (args.n_phases - 1) // 2)
        self.P = np.linalg.pinv(md.design(self.phases, args.period, self.nh))      # (n_coef, n_phase)
        self.box = None

    def __call__(self, raw):
        a = self.args
        raw = raw.astype(np.float32)
        H, W = self.split, raw.shape[-1]
        U = raw[:, :H]
        G = np.stack([cv2.warpAffine(f[H:2 * H], self.warp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan) for f in raw])
        bg = md.background(U, a.bg_margin)[:, None, None] if a.bg is None else a.bg
        U, G = U - bg, G - bg
        tot = np.nansum(U + G, axis=(1, 2))
        U, G, _ = md.normalize_frames(U, G)
        Ub, Gb = md.bin_stack(U, a.bin), md.bin_stack(G, a.bin)
        cU = np.tensordot(self.P, Ub, 1)
        cG = np.tensordot(self.P, Gb, 1)
        mask = np.isfinite(cU[0]) & np.isfinite(cG[0])
        mask &= (cU[0] > a.mask_thr * np.nanpercentile(cU[0], 99)) & (cG[0] > a.mask_thr * np.nanpercentile(cG[0], 99))
        mU, mG = np.where(mask, md.m1(cU), np.nan), np.where(mask, md.m1(cG), np.nan)

        if self.box is None:                                    # ROI fixed after the first cycle
            if a.roi_center is not None:
                cy, cx = a.roi_center
            else:
                cy, cx = ((np.array(ndi.center_of_mass(mask)) + 0.5) * a.bin).round().astype(int)
            h = a.roi // 2
            self.box = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
        u, g = (np.nansum(X[:, self.box[0], self.box[1]], axis=(1, 2)) for X in (U, G))
        return dict(mU=mU, mG=mG, mAvg=0.5 * (mU + mG), u=u, g=g, cu=self.P @ u, cg=self.P @ g,
                    tot=tot / np.median(tot), bg=float(np.mean(bg)))


# ------------------------------------------------------------------ display
class Display:
    def __init__(self, args, proc):
        self.args, self.proc = args, proc
        plt.ion()
        self.fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
        self.ax = ax
        self.ims = []
        for a, t in zip((ax[0, 0], ax[0, 1], ax[0, 2]), ("U m1", "G m1", "mean m1")):
            im = a.imshow(np.zeros((2, 2)), cmap="viridis", vmin=args.vmin, vmax=args.vmax)
            self.fig.colorbar(im, ax=a, fraction=0.03)
            a.set_title(t)
            a.set_xticks([])
            a.set_yticks([])
            self.ims.append(im)
        self.rects = []
        dense = np.linspace(0, args.period, 181)
        self.dense, self.Dd = dense, md.design(dense, args.period, proc.nh)
        a = ax[1, 0]
        self.pts = [a.plot([], [], "o", c=c)[0] for c in ("C0", "C3")]
        self.fits = [a.plot([], [], "-", c=c, label=lab)[0] for c, lab in (("C0", "U"), ("C3", "G"))]
        a.set_xlim(0, args.period)
        a.set_ylim(0, 2)
        a.set_xlabel("RF phase [deg]")
        a.set_ylabel("ROI intensity / mean")
        a.legend(loc="upper right")
        a.grid(alpha=0.3)
        a = ax[1, 1]
        self.bars = a.bar(proc.phases, np.ones(len(proc.phases)), width=args.phase_step * 0.8, color="0.5")
        a.axhline(1, c="k", lw=0.8)
        a.set_xticks(proc.phases)
        a.set_ylim(0, 2)
        a.set_xlabel("RF phase [deg]")
        a.set_title("U + G per frame / median (excitation)")
        a.grid(alpha=0.3)
        a = ax[1, 2]
        a.axis("off")
        self.text = a.text(0, 1, "", va="top", family="monospace", fontsize=11)
        self.history = []
        self.fig.tight_layout()

    def update(self, r, n, dt):
        a = self.args
        for im, m in zip(self.ims, (r["mU"], r["mG"], r["mAvg"])):
            im.set_data(m)
            im.set_extent((-0.5, m.shape[1] - 0.5, m.shape[0] - 0.5, -0.5))
        if not self.rects:
            b = self.proc.box
            for ax_ in self.ax[0]:
                self.rects.append(ax_.add_patch(plt.Rectangle((b[1].start / a.bin - 0.5, b[0].start / a.bin - 0.5),
                                                              a.roi / a.bin, a.roi / a.bin, fill=False, ec="w")))
        ph = self.proc.phases
        for pt, fit, y, c in zip(self.pts, self.fits, (r["u"], r["g"]), (r["cu"], r["cg"])):
            pt.set_data(ph, y / c[0])
            fit.set_data(self.dense, self.Dd @ c / c[0])
        for b, v in zip(self.bars, r["tot"]):
            b.set_height(v)
            b.set_color("C3" if abs(v - 1) > 0.2 else "0.5")
        mu, mg = md.m1(r["cu"]), md.m1(r["cg"])
        self.history = (self.history + [0.5 * (mu + mg)])[-20:]
        self.text.set_text(
            f"cycle {n}   ({dt:.1f} s)\n\n"
            f"ROI m1   U {mu:.3f}   G {mg:.3f}   mean {0.5 * (mu + mg):.3f}\n"
            f"map median mean m1  {np.nanmedian(r['mAvg']):.3f}\n"
            f"map 99th pct        {np.nanpercentile(r['mAvg'], 99):.3f}\n\n"
            f"U+G per frame  {np.array2string(r['tot'], precision=2)}\n"
            f"camera offset  {r['bg']:.1f}\n\n"
            f"last {len(self.history)} cycles ROI mean m1:\n  "
            + " ".join(f"{v:.2f}" for v in self.history[-10:]))
        self.fig.canvas.draw_idle()
        plt.pause(0.01)
        print(f"cycle {n}: ROI m1 U {mu:.3f} G {mg:.3f} mean {0.5 * (mu + mg):.3f}; "
              f"U+G per frame {np.round(r['tot'], 2)} ({dt:.1f} s)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", help="Micro-Manager config group that sets the phase")
    p.add_argument("--presets", nargs="+", help="one preset of --group per phase, in phase order")
    p.add_argument("--device", help="device whose property sets the phase")
    p.add_argument("--prop", help="property of --device that sets the phase")
    p.add_argument("--values", nargs="+", help="one property value per phase, in phase order")
    p.add_argument("--exposure", type=float, default=None, help="camera exposure [ms] (default: as set in MM)")
    p.add_argument("--file", type=Path, help="replay a saved stack instead of acquiring")
    p.add_argument("--loop", action="store_true", help="with --file: replay forever")
    p.add_argument("--reg", type=Path, default=rv.DEFAULT_REG, help="registration json (warp + split)")
    p.add_argument("--n-phases", type=int, default=6)
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    p.add_argument("--period", type=float, default=180.0, help="gating period in RF phase [deg]")
    p.add_argument("--harmonics", type=int, default=2)
    p.add_argument("--bin", type=int, default=16, help="spatial binning for the maps")
    p.add_argument("--roi", type=int, default=200, help="ROI side length [full-res px]")
    p.add_argument("--roi-center", type=int, nargs=2, default=None, metavar=("Y", "X"))
    p.add_argument("--mask-thr", type=float, default=0.3)
    p.add_argument("--bg", type=float, default=None, help="fixed camera offset (default: from image margins)")
    p.add_argument("--bg-margin", type=int, default=250)
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=0.6)
    p.add_argument("--max-cycles", type=int, default=None, help="stop after this many cycles")
    p.add_argument("--snapshot", type=Path, default=None, help="save the window to this png after each cycle")
    args = p.parse_args()
    if args.presets is not None:
        args.n_phases = len(args.presets)
    elif args.values is not None:
        args.n_phases = len(args.values)

    source = Replay(args) if args.file else Microscope(args)
    proc = Processor(args)
    disp = Display(args, proc)
    n = 0
    while plt.fignum_exists(disp.fig.number) and (args.max_cycles is None or n < args.max_cycles):
        t0 = time.perf_counter()
        raw = source.cycle()
        if raw is None:
            break
        n += 1
        disp.update(proc(raw), n, time.perf_counter() - t0)
        if args.snapshot is not None:
            disp.fig.savefig(args.snapshot, dpi=70)
    plt.ioff()
    if plt.fignum_exists(disp.fig.number):
        plt.show()


if __name__ == "__main__":
    main()
