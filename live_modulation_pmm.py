"""
live_modulation.py, but driving the hardware with pymmcore (MMCore loaded directly in Python)
instead of pycromanager's bridge to the Micro-Manager GUI. Frames come out of the circular buffer
as numpy arrays (one memcpy) instead of being serialised over a socket (~70 ms per full frame).

Micro-Manager must be CLOSED: this process owns the camera and COM12 itself. pymmcore's device
interface version must match the adapters of the MM install whose config is loaded (the 4th
number of the pymmcore version, e.g. 11.10.0.74.0 for Micro-Manager-2.0gamma_API_74_*).

At the moment, the Tucsen gen device adapter does not seem to work with API 75 
(even though there is one)

Modes (all phase control over serial to the 409C DDS, table from set_phases*.bsh, n rows):
    trig  camera in software-trigger mode, sequence started once: per cycle n software pulses ->
          exactly n frames (each steps the table). No per-cycle camera start, nothing untriggered
          fills the buffer, and the phase count can't drift. The table must wrap after n rows.
    seq  per cycle: reset the table to row 0, burst of n frames (sequence acquisition)
    snap  per cycle: reset the table to row 0, n snaps
    free  camera free-runs (table armed once); frame k is phase k mod n, the newest complete block
          is processed while the camera keeps running. Dropped frames (gaps in the buffer's
          ImageNumber tag, or an overflow) re-sync the table. The table must wrap after n rows.

Usage
-----
    python live_modulation_pmm.py --mode trig [--exposure 10]
    python live_modulation_pmm.py --mode trig --pulse-interval-ms 40   # overlap exposure with readout
    python live_modulation_pmm.py --mode free [--config <MM .cfg>] [--exposure 10]
    python live_modulation_pmm.py --mode seq
"""
import argparse
import atexit
import ctypes
import faulthandler
import os
import time
from pathlib import Path

import numpy as np

import live_modulation as lm
from live_modulation import T

DEFAULT_CFG_TEST = Path(r"C:\Program Files\Micro-Manager-2.0gamma_API_74\tucsen_novatech.cfg")
DEFAULT_CFG = Path(r"C:\Program Files\Micro-Manager-2.0gamma_API_73_20250530\nikon_tucsen_asixyz_novatech_obis.cfg")
DEFAULT_DEVICES_SEARCH_PATH = Path(r"C:\Program Files\Micro-Manager-2.0gamma_API_74")
DEFAULT_DEVICES_SEARCH_PATH_API73 = Path(r"C:\Program Files\Micro-Manager-2.0gamma_API_73_20250530")
# API 73 needs pymmcore 11.9.0.73.0

MAX_LOAD_TRIES = 3
MAX_RESYNCS = 3   # trig mode: consecutive failed re-syncs before giving up


def reset_sensor_roi(c, cam):
    """Full-sensor frame. The Tucsen adapter's clearROI() doesn't restore the GenICam Height
    (ImageFormatControl), so a smaller height sticks, even across sessions. Do what SamplePro needs by
    hand: offsets to 0 first (so the full size is allowed), then Height / Width to their maximum.
    Height, offsets etc. are read-only unless TriggerMode is FreeRunning, so switch to it if needed."""
    names = {n.lower().replace(" ", ""): n for n in c.getDevicePropertyNames(cam)}
    if c.isSequenceRunning():
        c.stopSequenceAcquisition()
    mode = c.getProperty(cam, "TriggerMode") if "triggermode" in names else None
    if mode is not None and mode != "FreeRunning":
        c.setProperty(cam, "TriggerMode", "FreeRunning")  # unlocks Height / Offset*

    def prop(key):  # e.g. "Height" or "ImageFormatControl Height", but not "HeightMax"
        hits = [n for k, n in names.items() if k.endswith(key.lower()) and "max" not in k]
        return min(hits, key=len) if hits else None

    done, locked = [], []
    for key, to_max in (("OffsetY", False), ("OffsetX", False), ("Height", True), ("Width", True)):
        name = prop(key)
        if name is None:
            continue
        if c.isPropertyReadOnly(cam, name):
            locked.append(f"{name}={c.getProperty(cam, name)}")
            continue
        if to_max:
            if c.hasPropertyLimits(cam, name):
                value = c.getPropertyUpperLimit(cam, name)
            else:
                allowed = [float(v) for v in c.getAllowedPropertyValues(cam, name)]
                if not allowed:
                    continue
                value = max(allowed)
        else:
            value = 0
        old = c.getProperty(cam, name)
        if float(old) != float(value):
            c.setProperty(cam, name, str(int(value)))
            done.append(f"{name} {old} -> {c.getProperty(cam, name)}")
    c.clearROI()
    if mode is not None and mode != "FreeRunning":
        c.setProperty(cam, "TriggerMode", mode)
    print(f"sensor reset to full frame {c.getImageWidth()} x {c.getImageHeight()}"
          + (f" ({', '.join(done)})" if done else ""))
    if locked:
        print(f"*** read-only, not reset: {', '.join(locked)} (reset Height in SamplePro if the frame is short)")

class PmmMicroscope:
    def __init__(self, args):
        import pymmcore  # only needed live

        mm_dir = str(args.mm_dir or args.config.parent)
        os.add_dll_directory(mm_dir)  # adapters' own dependencies (e.g. the Tucsen SDK) live there
        os.environ["PATH"] = mm_dir + os.pathsep + os.environ["PATH"]
        os.chdir(mm_dir)              # as the MM GUI; some SDKs look for files in the cwd
        # MMCore's LoadLibrary doesn't find the Tucsen SDK's dependency chain in mm_dir ("%1 is not a
        # valid Win32 application"); ctypes does. Once loaded, MMCore gets the already-loaded copies.
        for name in args.preload:
            path = os.path.join(mm_dir, name)
            if os.path.isfile(path):
                ctypes.WinDLL(path)
        self.pymmcore = pymmcore
        self.args = args
        # watchdog: a camera / serial call that never returns hangs silently; this prints where it's stuck
        faulthandler.dump_traceback_later(args.watchdog_s, repeat=True)
        if args.core == "plus":       # pymmcore-plus: a CMMCore subclass, same calls
            from pymmcore_plus import CMMCorePlus
            c = self.core = CMMCorePlus()
        else:
            c = self.core = pymmcore.CMMCore()
        if args.mm_log is not None:  # MMCore + device adapter messages (the Tucsen adapter logs its SDK calls)
            c.setPrimaryLogFile(str(args.mm_log), True)
            c.enableDebugLog(True)
            print(f"MMCore debug log: {args.mm_log}")
        c.setDeviceAdapterSearchPaths([mm_dir])
        print(f"{c.getVersionInfo()}, {c.getAPIVersionInfo()}; loading {args.config}")
        for tries in range(1, MAX_LOAD_TRIES + 1):
            try:
                c.loadSystemConfiguration(str(args.config))
                break
            except Exception as e:
                print(f"Failed, try {tries} of {MAX_LOAD_TRIES}: {' '.join(str(e).split())}")
                # a failed load leaves the devices that did load (e.g. COM12) in place, which makes the
                # next try fail on "already loaded"/"port in use"; start clean and give the camera a moment
                c.unloadAllDevices()
                if tries == MAX_LOAD_TRIES:
                    raise SystemExit(f"could not load {args.config} after {MAX_LOAD_TRIES} tries "
                                     "(is Micro-Manager, or a previous run, still holding the camera / COM12?)")
                time.sleep(2)
        self.closed = False
        atexit.register(self.close)   # also on errors during the rest of startup, which never reach run()
        c.setCircularBufferMemoryFootprint(args.buffer_mb)
        # start from the camera's normal free-running mode: a run that didn't exit cleanly can leave it
        # in Software trigger mode, where the ROI snap below can stall, and which would then be
        # "restored" on exit. FreeRunning is also what close() puts back (for MM live).
        cam = args.trigger_camera
        # The Tucsen adapter has hidden state that depends on the exact sequence of calls: extra
        # property reads / an early TriggerMode set here made later TriggerMode sets never reach the
        # camera (or pulses do nothing). So this startup follows, step for step, the sequence that
        # works in test_software_trigger.py --all: buffer, full-frame reset, ROI, trigger output, DDS,
        # FreeRunning, 1 s, Software, start - with no other reads or sets in between.
        self.normal_trigger_mode = "FreeRunning"
        reset_sensor_roi(c, cam)      # undo a height left small by an earlier ROI (adapter bug)
        if args.sensor_mode is not None:  # e.g. Speed: faster readout; only settable in FreeRunning
            c.setProperty(cam, "SensorOperationMode", args.sensor_mode)
        if args.cam_roi is not None:  # fewer rows: shorter readout + rolling start, fewer pixels to transfer
            # (no snap here: with the API 74 adapter a snap after the ROI change delivers a late,
            # untriggered frame once software triggering starts)
            c.setROI(*args.cam_roi)
        x, y, w, h = c.getROI()       # what the camera actually set (it may round to its ROI step)
        args.roi_offset = (x, y)      # the processor shifts the full-frame registration by this
        print(f"camera ROI x={x} y={y} w={w} h={h}" + ("" if args.cam_roi is None else f" (asked {args.cam_roi})"))
        split = lm.rv.load_warp(args.reg)[1]
        if not y < split < y + h:
            raise SystemExit(f"camera ROI rows {y}..{y + h} must straddle the registration's split row {split}")
        if args.exposure is not None:
            c.setExposure(args.exposure)
        c.setProperty(args.trigger_camera, "TriggerPort", args.trigger_port)
        c.setProperty(args.trigger_camera, "TriggerPortEnable", "On")
        self.block = None
        self.warned_no_number = False
        self.trig_mode = None
        self.laser = None
        if args.mode == "free":
            self.start_free()
        elif args.mode == "trig":
            self.start_trig()
        # once, after the camera / DDS setup (kept out of that call sequence); close() turns it off
        self.start_laser()

    # ---- cleans up tucsen
    def cleanup_cam(self):
        c, cam = self.core, self.args.trigger_camera
        c.setProperty(cam, "TriggerMode", "FreeRunning")

    # ---- 409C serial (as serialFunctions.bsh)
    def dds(self, cmd):
        """Send a command to the 409C and read lines until its status line ("OK" or "?x")."""
        port, c = self.args.dds_port, self.core
        c.setSerialPortCommand(port, cmd, "\r\n")
        lines = []
        while True:
            try:
                reply = c.getSerialPortAnswer(port, "\n").strip()
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
        """Stop the camera, disable stepping, table back to row 0, release the hardware.
        Runs once, from run()'s finally or atexit (startup errors never reach run()); every step is
        separate so one failure doesn't leave the camera open: the camera keeps its state across
        sessions, and a camera that was never closed properly ignores software pulses next time."""
        if getattr(self, "closed", True):
            return
        self.closed = True
        c, cam = self.core, self.args.trigger_camera
        faulthandler.cancel_dump_traceback_later()
        steps = [("laser off", self.stop_laser),  # first: the safety-relevant one
                 ("stop acquisition", lambda: c.isSequenceRunning() and c.stopSequenceAcquisition()),
                 ("DDS stepping off", lambda: [self.dds(cmd) for cmd in ("I d", "TSTOP", "TS 0")]),
                 ("TriggerMode", lambda: c.setProperty(cam, "TriggerMode", getattr(self, "normal_trigger_mode",
                                                                                   "FreeRunning"))),
                 ("trigger output off", lambda: c.setProperty(cam, "TriggerPortEnable", "Off")),
                 # leave the full frame for MM / SamplePro (the adapter won't restore the height itself)
                 ("full frame", lambda: reset_sensor_roi(c, cam)),
                 ("unload devices", c.unloadAllDevices)]
        for what, step in steps:
            try:
                step()
            except Exception as e:
                print(f"*** close: {what} failed: {' '.join(str(e).split())}")

    def to_stack(self, frames):
        """(n, H, W) float32 for the processor (the only per-frame conversion)."""
        if self.block is None:
            self.block = np.empty((len(frames),) + frames[0].shape, np.float32)
        for i, f in enumerate(frames):
            self.block[i] = f
        T.lap("to float")
        return self.block

    # ---- laser power (the one property in the --laser-group-name config group)
    def start_laser(self):
        c = self.core
        cfg_state_pow = c.getConfigGroupState(self.args.laser_power_group_name)
        sz = cfg_state_pow.size()
        if sz != 1:
            raise Exception(
                f"Group {self.args.laser_power_group_name} should have exactly one property! It has {sz}")
        setting = cfg_state_pow.getSetting(0)
        self.laser_power = (setting.getDeviceLabel(), setting.getPropertyName())  # for stop_laser()
        c.setProperty(*self.laser_power, self.args.laser_power)
        print(f"Laser {self.laser_power[0]} {self.laser_power[1]} = {c.getProperty(*self.laser_power)}")

        # now blank the laser on (digital enable)
        cfg_state_on = c.getConfigGroupState(self.args.laser_enable_group_name)
        sz = cfg_state_on.size()
        if sz != 1:
            raise Exception(
                f"Group {self.args.laser_enable_group_name} should have exactly one property! It has {sz}")
        setting = cfg_state_on.getSetting(0)
        self.laser_enable = (setting.getDeviceLabel(), setting.getPropertyName())  # for stop_laser()
        c.setProperty(*self.laser_enable, "1")
        print(f"Laser {self.laser_enable[0]} {self.laser_enable[1]} = {c.getProperty(*self.laser_enable)}")

    def stop_laser(self):
        """Power to 0 (or the property's lower limit); called first in close()."""
        if getattr(self, "laser_power", None) is None:
            return
        if getattr(self, "laser_enable", None) is None:
            return
        c = self.core
        # and turn power to 0
        off = c.getPropertyLowerLimit(*self.laser_power) if c.hasPropertyLimits(*self.laser_power) else 0
        c.setProperty(*self.laser_power, str(max(off, 0)))
        print(f"Laser {self.laser_power[0]} {self.laser_power[1]} = {c.getProperty(*self.laser_power)} (off)")
        # and turn state to 0 (disable)
        c.setProperty(*self.laser_enable, "0")
        print(f"Laser {self.laser_enable[0]} {self.laser_enable[1]} = {c.getProperty(*self.laser_enable)} (off)")

    # ---- seq / snap: one block per cycle
    def cycle(self):
        faulthandler.dump_traceback_later(self.args.watchdog_s, repeat=True)  # re-arm per cycle
        if self.args.mode == "free":
            return self.cycle_free()
        if self.args.mode == "trig":
            return self.cycle_trig()
        c, n = self.core, self.args.n_phases
        if c.isSequenceRunning():
            c.stopSequenceAcquisition()
        self.arm()
        exp_ms = c.getExposure()
        T.ideal = n * exp_ms / 1000
        frames = []
        if self.args.mode == "snap":
            for _ in range(n):
                c.snapImage()
                T.lap("snap")
                frames.append(c.getImage())
                T.lap("get_image")
            return self.to_stack(frames)

        c.clearCircularBuffer()
        c.startSequenceAcquisition(n, 0, True)
        T.lap("seq start")
        t_end = time.perf_counter() + 5 + n * exp_ms / 1000
        try:
            while len(frames) < n:
                if c.getRemainingImageCount() > 0:
                    T.lap("first frame wait" if not frames else "frame wait")
                    frames.append(c.popNextImage())
                    T.lap("pop")
                elif not c.isSequenceRunning():
                    raise RuntimeError(f"sequence stopped after {len(frames)}/{n} frames")
                elif time.perf_counter() > t_end:
                    raise TimeoutError(f"only {len(frames)}/{n} frames arrived")
                else:
                    time.sleep(0.001)
        finally:
            if c.isSequenceRunning():
                c.stopSequenceAcquisition()
            T.lap("seq stop")
        return self.to_stack(frames)

    # ---- software triggered: sequence started once, one frame per software pulse
    def pick(self, prop, wanted, avoid=()):
        """Allowed value of a camera property containing one of `wanted` (case-insensitive)."""
        cam = self.args.trigger_camera
        vals = list(self.core.getAllowedPropertyValues(cam, prop))
        print(f"{cam} {prop}: allowed {vals or '(any)'}, now {self.core.getProperty(cam, prop)!r}")
        for w in wanted:
            for v in vals:
                if w.lower() in v.lower() and v not in avoid:
                    return v
        return None

    def set_trigger_mode(self, mode, real_s=0.0005, bounce=False):
        """Set TriggerMode and make sure it reaches the camera. A set the camera receives takes
        ~2.5 ms (Micro-Manager GUI); the adapter sometimes only stores the value (returns in a few us), and the
        camera then starts in its old mode. In that case go through another mode (preferably
        Standard, which just waits for a hardware trigger) so the adapter sees a real change."""
        c, cam = self.core, self.args.trigger_camera

        def timed(value):
            t0 = time.perf_counter()
            c.setProperty(cam, "TriggerMode", value)
            return time.perf_counter() - t0

        dt = timed(mode)
        if dt >= real_s:
            return
        if not bounce:  # the working test never bounces; a repeat of the current mode is also this fast
            print(f"TriggerMode {mode!r} set returned in {1e3 * dt:.3f} ms (camera already in it, or skipped)")
            return
        allowed = list(c.getAllowedPropertyValues(cam, "TriggerMode"))
        others = sorted((m for m in allowed if m != mode), key=lambda m: m != "Standard")
        for other in others:
            timed(other)
            dt = timed(mode)
            if dt >= real_s:
                print(f"TriggerMode {mode!r}: adapter skipped the set, applied via {other!r} ({1e3 * dt:.0f} ms)")
                return
        print(f"*** TriggerMode {mode!r} set returns in {1e3 * dt:.2f} ms even via {others}: "
              "it may not be reaching the camera")

    def start_trig(self):
        """Camera waits for software pulses while a continuous sequence runs, so the camera never
        has to be (re)started per cycle and no untriggered frames fill the buffer."""
        c, a = self.core, self.args
        cam = a.trigger_camera
        if c.isSequenceRunning():
            c.stopSequenceAcquisition()
        if self.trig_mode is None:  # first start: the trigger values, without reading them from the camera
            # (pick()'s property reads at this point are part of what made the adapter ignore later
            # TriggerMode sets; the values are the ones this camera reports)
            self.trig_mode = a.trig_mode or "Software"
            self.pulse = a.trig_pulse or "TriggerSoftwarePulse"
            print(f"TriggerMode = {self.trig_mode!r}, pulse with TriggerSoftwarePulse = {self.pulse!r}")
        # Exactly the sequence of the working test_software_trigger.py / .bsh: FreeRunning, a 1 s pause,
        # Software, start. Setting Software right after stopping an acquisition (or without the pause)
        # came back in microseconds and never reached the camera.
        self.arm()
        self.set_trigger_mode(self.normal_trigger_mode)
        time.sleep(1.0)
        print(f"BEING SET TO  {self.trig_mode}")
        self.set_trigger_mode(self.trig_mode)
        # If the start still resets it to FreeRunning, set it again while running.
        c.clearCircularBuffer()
        c.startContinuousSequenceAcquisition(0)
        now = c.getProperty(cam, "TriggerMode")
        rearm = now != self.trig_mode  # free-running frames before the fix-up stepped the table
        if now != self.trig_mode:
            print(f"TriggerMode reset to {now!r} by starting acquisition: setting {self.trig_mode!r} while running")
            try:
                print(f"BEING SET TO  {self.trig_mode}")
                c.setProperty(cam, "TriggerMode", self.trig_mode)
            except Exception as e:
                c.stopSequenceAcquisition()
                raise SystemExit(f"TriggerMode can't be set while acquiring ({' '.join(str(e).split())}); "
                                 "software triggering with a camera ROI doesn't work here: try without --cam-roi, "
                                 "or --mode free with it")
            now = c.getProperty(cam, "TriggerMode")
            print("NOW IS:", now)
            if now != self.trig_mode:
                c.stopSequenceAcquisition()
                raise SystemExit(f"TriggerMode won't stay {self.trig_mode!r} once acquisition starts (reads {now!r})")
            c.clearCircularBuffer()  # frames from the free-running moment; the stray check below re-arms
        # No pulses yet, so no frames should arrive. The camera may still deliver a stray frame or two
        # when (re)starting, e.g. after an ROI change: drop those and re-arm the table (each one stepped
        # it). Frames that keep coming mean the camera is actually free-running.
        wait = max(0.3, 5 * c.getExposure() / 1000)
        for attempt in range(3):
            time.sleep(wait)
            stray = c.getRemainingImageCount()
            if stray == 0:
                break
            print(f"{stray} untriggered frame(s) at start: discarding, re-arming the phase table")
            c.clearCircularBuffer()
        else:
            c.stopSequenceAcquisition()
            raise SystemExit(f"frames keep arriving without a trigger: TriggerMode "
                             f"{c.getProperty(cam, 'TriggerMode')!r} isn't software triggering")
        if attempt or rearm:
            self.arm()
        T.lap("seq start")

    def fire_pulse(self):
        """One software trigger. The read first is a cheap precaution from one run where pulses never
        reached the camera (set returned in ~0.05 ms instead of ~2.2 ms); tests since then fire
        reliably with or without it."""
        c, cam = self.core, self.args.trigger_camera
        c.getProperty(cam, "TriggerSoftwarePulse")
        c.setProperty(cam, "TriggerSoftwarePulse", self.pulse)

    def resync_trig(self, why):
        """Restart the triggered sequence (table back to row 0) and retake the cycle; give up after a
        few in a row instead of looping forever with the window frozen."""
        c, cam = self.core, self.args.trigger_camera
        self.resyncs = getattr(self, "resyncs", 0) + 1
        state = (f"TriggerMode {c.getProperty(cam, 'TriggerMode')!r}, sequence running "
                 f"{c.isSequenceRunning()}, frames waiting {c.getRemainingImageCount()}")
        print(f"{why}: re-syncing the phase table ({self.resyncs}/{MAX_RESYNCS}; {state})")
        if self.resyncs >= MAX_RESYNCS:
            # set run mode back to internal trigger
            self.cleanup_cam()
            raise SystemExit(f"software trigger isn't producing frames ({state})")
        self.start_trig()
        return self.cycle_trig()

    def cycle_trig(self):
        frames = self._cycle_trig()
        self.resyncs = 0  # a good cycle
        return frames

    def _cycle_trig(self):
        if self.args.pulse_interval_ms > 0:
            return self.cycle_trig_paced()
        c, n, cam = self.core, self.args.n_phases, self.args.trigger_camera
        exp_ms = c.getExposure()
        frames = []
        for i in range(n):  # one pulse at a time: a pulse during exposure / readout could be dropped
            self.fire_pulse()
            T.lap("pulse")
            t_end = time.perf_counter() + 1 + 2 * exp_ms / 1000
            while c.getRemainingImageCount() == 0:
                if time.perf_counter() > t_end or not c.isSequenceRunning():
                    return self.resync_trig(f"*** no frame for pulse {i + 1}/{n}")
                time.sleep(0.0005)
            T.lap("frame wait")
            frames.append(c.popNextImage())
            T.lap("pop")
        if c.getRemainingImageCount() > 0:  # extra frames = untriggered frames: phases unknown
            return self.resync_trig(f"*** {c.getRemainingImageCount()} unexpected extra frames")
        T.ideal = n * exp_ms / 1000
        return self.to_stack(frames)

    def cycle_trig_paced(self):
        """Pulses on a fixed schedule (--pulse-interval-ms) instead of waiting for each frame, so the
        next exposure overlaps the previous frame's readout / transfer. A pulse the camera ignores
        makes no exposure, hence no trigger-out and no DDS step: frame k is still row k, it just
        costs another pulse."""
        c, n, cam = self.core, self.args.n_phases, self.args.trigger_camera
        exp_ms, dt = c.getExposure(), self.args.pulse_interval_ms / 1000
        frames, pulses = [], 0
        t_next = time.perf_counter()
        t_last_frame = t_next
        settle = max(0.3, 4 * (exp_ms / 1000 + dt))          # no frame for this long: pulses were dropped
        while len(frames) < n:
            now = time.perf_counter()
            if pulses < n and now >= t_next:
                self.fire_pulse()
                pulses += 1
                t_next = now + dt
                T.lap("pulse")
            got = 0
            while c.getRemainingImageCount() > 0 and len(frames) < n:
                frames.append(c.popNextImage())
                got += 1
            if got:
                t_last_frame = time.perf_counter()
                T.lap("pop", got)
            elif pulses >= n and time.perf_counter() - t_last_frame > settle:
                if pulses >= 2 * n or not c.isSequenceRunning():
                    return self.resync_trig(f"*** only {len(frames)}/{n} frames for {pulses} pulses")
                self.fire_pulse()  # replace a dropped pulse
                pulses += 1
                t_last_frame = time.perf_counter()
                T.note = f"{pulses - n} pulse(s) dropped: try a longer --pulse-interval-ms"
                T.lap("pulse")
            else:
                time.sleep(0.0005)
                T.lap("frame wait")
        if pulses > n:                         # a "dropped" pulse may just have been late: catch its frame
            time.sleep(settle / 4)
            if c.getRemainingImageCount() > 0:
                return self.resync_trig(f"*** {c.getRemainingImageCount()} extra frames")
            T.lap("frame wait")
        T.ideal = n * exp_ms / 1000
        return self.to_stack(frames)

    # ---- free running
    def start_free(self):
        """Arm the table at row 0, then start the camera: frame k is phase row k mod n."""
        c = self.core
        if c.isSequenceRunning():
            c.stopSequenceAcquisition()
        self.arm()
        c.clearCircularBuffer()
        c.startContinuousSequenceAcquisition(0)
        self.count, self.last_number = 0, None
        self.cur, self.done, self.blocks = [None] * self.args.n_phases, None, 0  # blocks from before the re-sync are gone
        T.lap("seq start")

    def pop_counted(self):
        """Pop one frame; False if frames were lost since the last one (phase count is broken)."""
        md = self.pymmcore.Metadata()
        img = self.core.popNextImageMD(md)
        number = None
        if md.HasTag("ImageNumber"):
            number = int(md.GetSingleTag("ImageNumber").GetValue())
        elif not self.warned_no_number:
            print("*** frames carry no ImageNumber tag: dropped frames are only caught by overflow")
            self.warned_no_number = True
        ok = number is None or self.last_number is None or number == self.last_number + 1
        self.last_number = number
        k = self.count % self.args.n_phases
        self.cur[k] = img   # keep the array itself (popNextImage already copied it out of the buffer)
        self.count += 1
        if k == self.args.n_phases - 1:
            self.done, self.cur = self.cur, [None] * self.args.n_phases
            self.blocks += 1
        return ok

    def cycle_free(self):
        """Pop everything waiting (every frame must be counted) and return the newest complete block."""
        c, n = self.core, self.args.n_phases
        self.blocks, popped = 0, 0
        exp_ms = c.getExposure()
        t_end = time.perf_counter() + 5 + 2 * n * exp_ms / 1000
        while True:
            lost = c.isBufferOverflowed() or not c.isSequenceRunning()
            t_drain = time.perf_counter() + 1.0  # don't chase a camera that is faster than popping
            drained = 0
            while not lost and c.getRemainingImageCount() > 0 and (not self.blocks or time.perf_counter() < t_drain):
                lost = not self.pop_counted()
                drained += 1
            popped += drained
            T.lap("pop", drained)
            if lost:
                print(f"*** frames lost after {self.count} frames: re-syncing the phase table")
                self.start_free()
                t_end = time.perf_counter() + 5 + 2 * n * exp_ms / 1000
                continue
            if self.blocks:
                break
            if time.perf_counter() > t_end:
                raise TimeoutError(f"no complete {n}-frame block ({self.count} frames so far)")
            time.sleep(0.001)
            T.lap("frame wait")
        T.note = f"{popped} frames popped, {(self.blocks - 1) * n} skipped" if self.blocks > 1 else ""
        T.ideal = n * exp_ms / 1000
        return self.to_stack(self.done)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("trig", "free", "seq", "snap"), default="free")
    p.add_argument("--trig-mode", default=None,
                   help="trig: camera TriggerMode value for software triggering (default: the one containing 'Software')")
    p.add_argument("--trig-pulse", default=None,
                   help="trig: value written to TriggerSoftwarePulse to fire one frame (default: picked from its allowed values)")
    # p.add_argument("--dds-port", default="COM12", help="409C DDS serial port")
    p.add_argument("--laser-power", default="10.0", help="laser power in mW")
    p.add_argument("--laser-power-group-name", default="640 power", help="laser power in mW")
    p.add_argument("--laser-enable-group-name", default="640 enable", help="Name of MicroManager group used to control laser power")
    p.add_argument("--core", choices=("pymmcore", "plus"), default="pymmcore",
                   help="MMCore wrapper: plain pymmcore, or pymmcore-plus's CMMCorePlus")
    p.add_argument("--mm-log", type=Path, default=None,
                   help="write MMCore's debug log (incl. the Tucsen adapter's SDK calls / errors) to this file")
    p.add_argument("--watchdog-s", type=float, default=30,
                   help="print a stack trace if startup or one cycle takes longer than this (a hung camera call)")
    p.add_argument("--sensor-mode", default=None,
                   help="camera SensorOperationMode, e.g. Speed (faster readout) / Dynamic / Sensitivity "
                        "(default: leave as is)")
    p.add_argument("--cam-roi", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"),
                   help="camera ROI in full-frame pixels, e.g. 400 400 2400 2400 (must contain both views); "
                        "the registration is shifted to match")
    p.add_argument("--pulse-interval-ms", type=float, default=0,
                   help="trig: fire pulses this far apart without waiting for each frame (0 = wait for each); "
                        "start near exposure + readout and shorten until '[pulse(s) dropped]' appears")
    p.add_argument("--config", type=Path, default=DEFAULT_CFG, help="Micro-Manager hardware config (.cfg)")
    p.add_argument("--mm-dir", type=Path, default=DEFAULT_DEVICES_SEARCH_PATH_API73,
                   help="Micro-Manager install with the device adapters (must match pymmcore's device API)")
    p.add_argument("--preload", nargs="*", default=["TUCam.dll", "mmgr_dal_Tucsen_Gen_x64.dll"],
                   help="DLLs in --mm-dir to load before MMCore loads the adapters")
    p.add_argument("--buffer-mb", type=int, default=2000, help="circular buffer size [MB]")
    lm.add_args(p)
    args = p.parse_args()
    for k in ("config", "mm_dir", "reg", "file", "snapshot", "mm_log"):  # absolute before chdir to the MM folder
        if getattr(args, k) is not None:
            setattr(args, k, getattr(args, k).resolve())
    # meat of code in the old implementation that uses pycromanager
    lm.run(args, lm.Replay(args) if args.file else PmmMicroscope(args))


if __name__ == "__main__":
    main()
