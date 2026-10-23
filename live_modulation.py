"""
Live modulation-depth maps from the microscope via pycromanager (or a saved stack, for testing).

Each cycle takes one frame per RF phase (0, 30, ..., 150 deg by default), then like modulation.py
but binned early for speed: split the two views at the registration's split row, warp the bottom
view (G) onto the top (U) with the registration json, bin, subtract the camera offset (image
margins), normalise every bin so U + G is phase-independent, and fit DC + harmonics per bin
(pseudo-inverse, precomputed). Shown: m1 maps of U, G and their mean on a fixed colour scale, the
central-ROI traces with fits, and the per-frame U + G (excitation stability). The ROI (whole bins)
is fixed after the first cycle with signal.

Phase stepping (Micro-Manager must be running with the pycromanager server enabled): all phase
control is over serial to the 409C DDS. Program the table first (set_phases.bsh /
set_phases_multichannel.bsh, --n-phases rows). Each cycle the table is reset to row 0 and a burst
of n frames is taken; each frame's trigger-out advances the table one row. On exit the stepping is
disabled again.

Usage
-----
    python live_modulation.py [--dds-port COM12] --n-phases 6 --phase-step 30 [--exposure 100]
    python live_modulation.py --file <acquisition folder or .ome.tif> [--loop]   # replay, no microscope
"""
import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast
from mmcore_types import CMMCore  # not perfect typing file to help autocompletes

import cv2

# matplotlib's QtAgg tries PyQt6 first; here the pip --user PyQt6 picks up conda's Qt6 DLLs and fails
# half-way, after which PySide6 (conda, matching those DLLs) can't load either. Must be set before
# matplotlib touches Qt.
os.environ.setdefault("QT_API", "pyside6")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import tifffile
from scipy import ndimage as ndi

import modulation as md
import register_views as rv


# ------------------------------------------------------------------- timing
class Timer:
    """Lap timer: lap(name) charges the time since the previous lap to name, so every part of a
    cycle is accounted for. Repeated names (per frame, per DDS command) accumulate and are reported
    as "total (count x mean)"; count says how many items the lap covered (e.g. frames popped)."""

    def __init__(self):
        self.start()

    def start(self):
        self.t, self.n, self.t0, self.ideal, self.note = {}, {}, time.perf_counter(), None, ""
        self.last = self.t0

    def lap(self, name, count=1):
        now = time.perf_counter()
        self.t[name] = self.t.get(name, 0.0) + now - self.last
        self.n[name] = self.n.get(name, 0) + count
        self.last = now

    def report(self, n):
        total = self.last - self.t0

        def fmt(k, v):
            c = self.n[k]
            return f"{k} {1e3 * v:.0f}" + (f" ({c}x {1e3 * v / c:.1f})" if c > 1 else "")
        parts = " | ".join(fmt(k, v) for k, v in self.t.items())
        ideal = "" if self.ideal is None else f" (exposures alone {1e3 * self.ideal:.0f})"
        print(f"cycle {n} timing [ms]: {parts} | total {1e3 * total:.0f}{ideal}"
              + (f"  [{self.note}]" if self.note else ""))


T = Timer()


# ------------------------------------------------------------------ sources
class Microscope:
    """One frame per phase from Micro-Manager; the camera trigger-out steps the 409C DDS table."""

    def __init__(self, args):
        from pycromanager import Core  # only needed live
        self.core = cast("CMMCore", Core())
        self.args = args
        self.stop_running_sequence()
        if args.exposure is not None:
            self.core.set_exposure(args.exposure)
        self.core.set_property(args.trigger_camera, "TriggerPort", args.trigger_port)
        self.core.set_property(args.trigger_camera, "TriggerPortEnable", "On")
        self.frames = None  # float32 (n, H, W), reused every cycle

    def stop_running_sequence(self):
        """MM live mode (or a run that didn't clean up) leaves a sequence running, which blocks the
        bursts; its frames would also step the DDS table."""
        if not self.core.is_sequence_running():
            return
        print("sequence acquisition already running (MM live mode?): stopping it")
        try:  # turn live off through the GUI so it doesn't restart / complain
            from pycromanager import Studio
            Studio().live().set_live_mode_on(False)
        except Exception:
            pass
        if self.core.is_sequence_running():
            self.core.stop_sequence_acquisition()

    # ---- 409C serial (as serialFunctions.bsh)
    def dds(self, cmd):
        """Send a command to the 409C and read lines until its status line ("OK" or "?x")."""
        port, c = self.args.dds_port, self.core
        c.set_serial_port_command(port, cmd, "\r\n")
        lines = []
        while True:
            try:
                reply = str(c.get_serial_port_answer(port, "\n")).strip()
            except Exception:
                break  # timeout: nothing more to read
            if not reply:
                continue
            lines.append(reply)
            if reply == "OK" or reply.startswith("?"):
                break
        if lines and lines[-1].startswith("?"):
            print(f"*** 409C rejected '{cmd}': {' / '.join(lines)}")
        return lines

    def arm(self):
        """Table back to row 0, stepping on each camera trigger (TSTOP is needed before I s)."""
        for cmd in ("I d", "TSTOP", "TS 0", "I s"):
            self.dds(cmd)
            T.lap(f"dds '{cmd}'")

    def close(self):
        """Stop the camera, disable stepping, table back to row 0 (as disable_phase_stepping_live.bsh)."""
        if self.core.is_sequence_running():
            self.core.stop_sequence_acquisition()
        for cmd in ("I d", "TSTOP", "TS 0"):
            self.dds(cmd)
        self.core.set_property(self.args.trigger_camera, "TriggerPortEnable", "Off")

    def cycle(self):
        """One burst of n frames; each frame's trigger-out advances the table one row.

        Not free-running: every frame has to be popped (transferred to Python, ~70 ms full-frame)
        to know its phase, which is far slower than the camera at short exposures, and on overflow
        continuous mode silently clears the buffer, scrambling the phase count."""
        self.stop_running_sequence()  # e.g. live switched on in MM mid-run
        self.arm()  # re-sync every cycle so a stray trigger (e.g. MM live) can't shift the phases
        c, n = self.core, self.args.n_phases
        exp_ms = c.get_exposure()
        poll_s = min(0.5 * exp_ms, 20) / 1000
        if self.frames is None:
            self.frames = np.empty((n, c.get_image_height(), c.get_image_width()), np.float32)
        T.lap("core queries")

        if self.args.mode == "snap":  # one software-triggered frame at a time, as aquire_phases.bsh
            for i in range(n):
                c.snap_image()
                T.lap("snap")
                pix = np.asarray(c.get_image())
                T.lap("get_image (transfer)")
                self.frames[i] = pix.reshape(self.frames.shape[1:])
                T.lap("copy")
            T.ideal = n * exp_ms / 1000
            return self.frames

        c.clear_circular_buffer()
        c.start_sequence_acquisition(n, 0, True)
        T.lap("seq start")
        try:
            i, t_end = 0, time.perf_counter() + 5 + n * exp_ms / 1000
            while i < n:
                if c.get_remaining_image_count() > 0:
                    T.lap("first frame wait" if i == 0 else "frame wait")
                    pix = np.asarray(c.pop_next_image())  # skips the tag JSON
                    T.lap("pop (transfer)")
                    self.frames[i] = pix.reshape(self.frames.shape[1:])  # copy + to float in one pass
                    T.lap("copy")
                    i += 1
                elif not c.is_sequence_running():
                    raise RuntimeError(f"sequence stopped after {i}/{n} frames")
                elif time.perf_counter() > t_end:
                    raise TimeoutError(f"only {i}/{n} frames arrived")
                else:
                    time.sleep(poll_s)
        finally:
            if c.is_sequence_running():
                c.stop_sequence_acquisition()
            T.lap("seq stop")
        T.ideal = n * exp_ms / 1000
        return self.frames


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

    def close(self):
        pass


# --------------------------------------------------------------- processing
class Processor:
    def __init__(self, args):
        self.args = args
        self.warp, self.split = rv.load_warp(args.reg)
        self.phases = np.arange(args.n_phases) * args.phase_step
        self.nh = min(args.harmonics, (args.n_phases - 1) // 2)
        self.P = np.linalg.pinv(md.design(self.phases, args.period, self.nh))      # (n_coef, n_phase)
        self.box = None
        # camera ROI at (x0, y0): ROI pixel (x, y) is full-frame (x + x0, y + y0). The registration maps
        # full-frame U coords to G coords relative to the split row, so in ROI coords the split moves up
        # by y0 and only the warp's offset changes: t' = t + A (x0, y0) - (x0, 0)
        x0, y0 = args.roi_offset
        A, t = self.warp[:, :2], self.warp[:, 2] + self.warp[:, :2] @ [x0, y0] - [x0, 0]
        self.warp = np.hstack([A, t[:, None]])
        self.split -= y0
        # the registration in binned-pixel units: binned pixel i is centred on full-res b*i + (b-1)/2
        b, c = args.bin, (args.bin - 1) / 2
        self.warp_b = np.hstack([A, ((A @ [c, c] + t - c) / b)[:, None]])

    def __call__(self, raw):
        a, b = self.args, self.args.bin
        raw = raw.astype(np.float32, copy=False)  # live frames already are
        H, W = self.split, raw.shape[-1]
        Hb, Wb = H // b, W // b
        gH = min(raw.shape[1] - H, H) // b                      # G rows available below the split
        T.lap("proc: to float")
        # bin both views first (INTER_AREA at an integer factor = exact block mean), then warp the small
        # binned G: ~6x faster than warping full-res, same result to <0.1%. Offset subtraction commutes
        # with binning, and the normalisation only has to remove per-frame excitation changes.
        Ub = np.stack([cv2.resize(f[:Hb * b, :Wb * b], (Wb, Hb), interpolation=cv2.INTER_AREA) for f in raw])
        Gs = [cv2.resize(f[H:H + gH * b, :Wb * b], (Wb, gH), interpolation=cv2.INTER_AREA) for f in raw]
        T.lap("proc: bin")
        Gb = np.stack([cv2.warpAffine(g, self.warp_b, (Wb, Hb), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan) for g in Gs])
        T.lap("proc: warp")
        if a.bg is None:  # camera offset per frame from the unilluminated left/right margins of U
            m = max(a.bg_margin // a.bin, 1)
            bg = np.median(np.concatenate([Ub[:, :, :m], Ub[:, :, -m:]], axis=2), axis=(1, 2))[:, None, None]
        else:
            bg = a.bg
        Ub, Gb = Ub - bg, Gb - bg
        tot = np.nansum(Ub + Gb, axis=(1, 2))
        T.lap("proc: offset")
        Ub, Gb, _ = md.normalize_frames(Ub, Gb)
        T.lap("proc: normalise")
        cU = np.tensordot(self.P, Ub, 1)
        cG = np.tensordot(self.P, Gb, 1)
        mask = np.isfinite(cU[0]) & np.isfinite(cG[0])
        mask &= (cU[0] > a.mask_thr * np.nanpercentile(cU[0], 99)) & (cG[0] > a.mask_thr * np.nanpercentile(cG[0], 99))
        mU, mG = np.where(mask, md.m1(cU), np.nan), np.where(mask, md.m1(cG), np.nan)
        T.lap("proc: fit+mask")

        box = self.box                                          # in binned pixels
        if box is None:                                         # ROI fixed after the first cycle with signal
            if a.roi_center is not None:
                cy, cx = np.array(a.roi_center) // a.bin
            elif mask.any():
                cy, cx = np.array(ndi.center_of_mass(mask)).round().astype(int)
            else:                                               # no signal (e.g. excitation off): image centre
                cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
            rb = max(round(a.roi / a.bin), 1)
            y0, x0 = max(cy - rb // 2, 0), max(cx - rb // 2, 0)
            box = (slice(y0, y0 + rb), slice(x0, x0 + rb))
            if a.roi_center is not None or mask.any():
                self.box = box
        u, g = (np.nansum(X[:, box[0], box[1]], axis=(1, 2)) for X in (Ub, Gb))
        T.lap("proc: roi")
        npx = (box[0].stop - box[0].start) * (box[1].stop - box[1].start)
        return dict(mU=mU, mG=mG, mAvg=0.5 * (mU + mG), u=u, g=g, cu=self.P @ u, cg=self.P @ g,
                    tot=tot / np.median(tot), bg=float(np.mean(bg)), box=box,
                    mask_frac=float(mask.mean()), dc=(np.mean(u) / npx, np.mean(g) / npx))  # counts/px above offset


# ------------------------------------------------------------------ display
class Display:
    def __init__(self, args, proc):
        self.args, self.proc = args, proc
        if args.backend:  # QtAgg blits much faster than the default TkAgg (QT_API: see the imports)
            try:
                plt.switch_backend(args.backend)
            except Exception as e:
                print(f"backend {args.backend} unavailable ({e}); using {plt.get_backend()}")
        plt.ion()
        self.fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
        self.ax = ax
        self.ims = []
        for a, t in zip((ax[0, 0], ax[0, 1], ax[0, 2]), ("U m1", "G m1", "mean m1")):
            im = a.imshow(np.zeros((2, 2)), cmap="viridis", vmin=args.vmin, vmax=args.vmax,
                          interpolation="nearest")  # bins are pixels anyway; cheapest resampling
            self.fig.colorbar(im, ax=a, fraction=0.03)
            a.set_title(t)
            a.set_xticks([])
            a.set_yticks([])
            self.ims.append(im)
        self.rects = [a.add_patch(plt.Rectangle((0, 0), 0, 0, fill=False, ec="w")) for a in ax[0]]
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
        self.text = a.text(0, 1, "", va="top", family="monospace", fontsize=13)
        self.fig.tight_layout()

        # blitting: everything that changes is "animated", i.e. left out of a normal draw. A full draw
        # (first frame, resize) caches the static background; each update restores it and redraws
        # only these artists.
        self.dynamic = self.ims + self.rects + self.pts + self.fits + list(self.bars) + [self.text]
        for art in self.dynamic:
            art.set_animated(True)
        self.background, self.shape, self.saving = None, None, False
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)

    def save(self, path):
        """savefig skips animated artists, so un-animate them for the save."""
        self.saving = True
        for art in self.dynamic:
            art.set_animated(False)
        try:
            self.fig.savefig(path, dpi=70)
        finally:
            for art in self.dynamic:
                art.set_animated(True)
            self.saving = False

    def _on_draw(self, event):
        if self.saving:
            return
        cv = self.fig.canvas
        if getattr(cv, "supports_blit", False):
            self.background = cv.copy_from_bbox(self.fig.bbox)
        for art in self.dynamic:
            self.fig.draw_artist(art)

    def redraw(self):
        cv = self.fig.canvas
        if self.background is None:
            cv.draw()                     # full draw; _on_draw caches the background
        else:
            cv.restore_region(self.background)
            for art in self.dynamic:
                self.fig.draw_artist(art)
            for ax in self.ax.flat:       # push only the panels to the screen, not titles/colorbars/margins
                cv.blit(ax.bbox)
        cv.flush_events()                 # keep the window responsive (no plt.pause sleep)

    def update(self, r, n, dt):
        for im, m in zip(self.ims, (r["mU"], r["mG"], r["mAvg"])):
            im.set_data(m)
        if r["mU"].shape != self.shape:  # first frame: axes limits change, background must be redrawn
            self.shape = r["mU"].shape
            for im in self.ims:
                im.set_extent((-0.5, self.shape[1] - 0.5, self.shape[0] - 0.5, -0.5))
            self.background = None
        b = r["box"]
        xy = (b[1].start - 0.5, b[0].start - 0.5)
        for rect in self.rects:                                 # moves until the ROI is fixed
            rect.set_xy(xy)
            rect.set_width(b[1].stop - b[1].start)
            rect.set_height(b[0].stop - b[0].start)
        ph = self.proc.phases
        for pt, fit, y, c in zip(self.pts, self.fits, (r["u"], r["g"]), (r["cu"], r["cg"])):
            pt.set_data(ph, y / c[0])
            fit.set_data(self.dense, self.Dd @ c / c[0])
        for b, v in zip(self.bars, r["tot"]):
            b.set_height(v)
            b.set_color("C3" if abs(v - 1) > 0.2 else "0.5")
        mu, mg = md.m1(r["cu"]), md.m1(r["cg"])
        # matplotlib text costs ~0.25 ms per character to draw, so the panel keeps only the key numbers;
        # the full stats go to the console
        self.text.set_text(
            f"cycle {n}  {dt:.2f} s\n"
            f"ROI m1  U {mu:.3f}  G {mg:.3f}\n"
            f"mean {0.5 * (mu + mg):.3f}  map {np.nanmedian(r['mAvg']):.3f}\n"
            f"signal U {r['dc'][0]:.0f} G {r['dc'][1]:.0f}, {100 * r['mask_frac']:.0f}% shown")
        T.lap("disp: update artists")
        self.redraw()
        T.lap("disp: draw")
        print(f"cycle {n}: ROI m1 U {mu:.3f} G {mg:.3f} mean {0.5 * (mu + mg):.3f}; "
              f"map median {np.nanmedian(r['mAvg']):.3f}, 99th pct {np.nanpercentile(r['mAvg'], 99):.3f}; "
              f"ROI signal above offset U {r['dc'][0]:.1f} G {r['dc'][1]:.1f} counts/px; "
              f"{100 * r['mask_frac']:.1f}% of bins shown; offset {r['bg']:.1f}; "
              f"U+G per frame {np.round(r['tot'], 2)} ({dt:.2f} s)")


def add_args(p):
    """Arguments shared with live_modulation_pmm.py."""
    p.add_argument("--dds-port", default="COM12", help="409C DDS serial port")
    p.add_argument("--backend", default="QtAgg", help="matplotlib backend for the live window ('' = default)")
    p.add_argument("--trigger-camera", default="TUCamGen", help="camera device with the trigger output")
    p.add_argument("--trigger-port", default="Port2", help="camera TriggerPort wired to the 409C")
    p.add_argument("--exposure", type=float, default=None, help="camera exposure [ms] (default: as set in MM)")
    p.add_argument("--file", type=Path, help="replay a saved stack instead of acquiring")
    p.add_argument("--loop", action="store_true", help="with --file: replay forever")
    p.add_argument("--reg", type=Path, default=Path(r"C:\Users\Selvin Lab\Documents\MicroManager\scripts\bead_registration.json"), help="registration json (warp + split)")
    p.add_argument("--n-phases", type=int, default=6, help="rows in the DDS table (as in set_phases.bsh)")
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    p.add_argument("--period", type=float, default=180.0, help="gating period in RF phase [deg]")
    p.add_argument("--harmonics", type=int, default=2)
    p.add_argument("--bin", type=int, default=16, help="spatial binning for the maps")
    p.add_argument("--roi", type=int, default=200, help="ROI side length [full-res px]")
    p.add_argument("--roi-center", type=int, nargs=2, default=None, metavar=("Y", "X"),
                   help="analysis ROI centre [px of the (camera-ROI) frame]")
    p.add_argument("--roi-offset", type=int, nargs=2, default=(0, 0), metavar=("X", "Y"),
                   help="frames are a camera ROI starting at this full-frame pixel (registration is full-frame); "
                        "set automatically by live_modulation_pmm.py --cam-roi")
    p.add_argument("--mask-thr", type=float, default=0.3)
    p.add_argument("--bg", type=float, default=None, help="fixed camera offset (default: from image margins)")
    p.add_argument("--bg-margin", type=int, default=250)
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=0.8)
    p.add_argument("--max-cycles", type=int, default=None, help="stop after this many cycles")
    p.add_argument("--snapshot", type=Path, default=None, help="save the window to this png after each cycle")


def run(args, source):
    """Acquire / process / display loop; source has cycle() -> (n, H, W) frames or None, and close()."""
    proc = Processor(args)
    disp = Display(args, proc)
    n = 0
    try:
        while plt.fignum_exists(disp.fig.number) and (args.max_cycles is None or n < args.max_cycles):
            T.start()
            raw = source.cycle()
            if raw is None:
                break
            n += 1
            r = proc(raw)
            disp.update(r, n, time.perf_counter() - T.t0)
            if args.snapshot is not None:
                disp.save(args.snapshot)
                T.lap("snapshot png")
            T.report(n)
    finally:
        source.close()  # disarm the DDS stepping even on Ctrl-C / errors
    plt.ioff()
    if plt.fignum_exists(disp.fig.number):
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("seq", "snap"), default="seq",
                   help="per cycle: one n-frame sequence acquisition, or n separate snaps")
    add_args(p)
    args = p.parse_args()
    run(args, Replay(args) if args.file else Microscope(args))


if __name__ == "__main__":
    main()
