"""
Shared pieces of the pymmcore scripts: loading the hardware without the Micro-Manager GUI, and the
409C DDS serial helpers (serialFunctions.bsh).

Micro-Manager must be CLOSED: these scripts open the camera and COM12 themselves. pymmcore's device
interface must match the adapters (eoflim-pmm env: pymmcore 11.9.0.73.0 with the API_73 install).
Loading follows live_modulation_pmm.py: the adapter folder on the DLL path and as working folder, and
the Tucsen SDK + adapter preloaded with ctypes (MMCore can't resolve their dependencies itself).
"""
import argparse
import atexit
import ctypes
import os
import time
from pathlib import Path

DEFAULT_CFG = Path(r"C:\Program Files\Micro-Manager-2.0gamma_API_73_20250530\nikon_tucsen_asixyz_novatech_obis.cfg")
DEFAULT_MM_DIR = Path(r"C:\Program Files\Micro-Manager-2.0gamma_API_73_20250530")
PRELOAD = ("TUCam.dll", "mmgr_dal_Tucsen_Gen_x64.dll")
MAX_LOAD_TRIES = 3


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--config", type=Path, default=DEFAULT_CFG, help="Micro-Manager hardware config (.cfg)")
    p.add_argument("--mm-dir", type=Path, default=DEFAULT_MM_DIR,
                   help="Micro-Manager install with the device adapters (must match pymmcore's device API)")
    p.add_argument("--dds-port", default="COM12", help="409C DDS serial port")
    p.add_argument("--mm-log", type=Path, default=None, help="write MMCore's debug log to this file")


def load_core(args):
    """Load the hardware config with pymmcore; devices are unloaded again at exit."""
    import pymmcore

    for k in ("config", "mm_dir", "mm_log"):  # absolute before the chdir below
        if getattr(args, k, None) is not None:
            setattr(args, k, Path(getattr(args, k)).resolve())
    mm_dir = str(args.mm_dir)
    os.add_dll_directory(mm_dir)
    os.environ["PATH"] = mm_dir + os.pathsep + os.environ["PATH"]
    os.chdir(mm_dir)  # as the MM GUI
    for name in PRELOAD:
        path = os.path.join(mm_dir, name)
        if os.path.isfile(path):
            ctypes.WinDLL(path)

    c = pymmcore.CMMCore()
    if args.mm_log is not None:
        c.setPrimaryLogFile(str(args.mm_log), True)
        c.enableDebugLog(True)
    c.setDeviceAdapterSearchPaths([mm_dir])
    print(f"{c.getVersionInfo()}, {c.getAPIVersionInfo()}; loading {args.config}")
    for tries in range(1, MAX_LOAD_TRIES + 1):
        try:
            c.loadSystemConfiguration(str(args.config))
            break
        except Exception as e:
            print(f"Failed, try {tries} of {MAX_LOAD_TRIES}: {' '.join(str(e).split())}")
            c.unloadAllDevices()  # a failed load leaves the devices that did load in place
            if tries == MAX_LOAD_TRIES:
                raise SystemExit(f"could not load {args.config} (is Micro-Manager still holding the camera / COM12?)")
            time.sleep(2)
    atexit.register(c.unloadAllDevices)
    return c


# ---- Tucsen software triggering (copied from live_modulation_pmm.py, which has the reasoning)
# The adapter has hidden state that depends on the exact call sequence: keep to reset_sensor_roi ->
# exposure / trigger output -> start_software_trigger, with no extra camera reads or sets in between.
def reset_sensor_roi(c, cam):
    """Full-sensor frame. clearROI() doesn't restore the GenICam Height, so set offsets to 0 and
    Height / Width to their maximum by hand; they are read-only unless TriggerMode is FreeRunning."""
    names = {n.lower().replace(" ", ""): n for n in c.getDevicePropertyNames(cam)}
    if c.isSequenceRunning():
        c.stopSequenceAcquisition()
    mode = c.getProperty(cam, "TriggerMode") if "triggermode" in names else None
    if mode is not None and mode != "FreeRunning":
        c.setProperty(cam, "TriggerMode", "FreeRunning")  # unlocks Height / Offset*

    def prop(key):  # e.g. "Height" or "ImageFormatControl Height", but not "HeightMax"
        hits = [n for k, n in names.items() if k.endswith(key.lower()) and "max" not in k]
        return min(hits, key=len) if hits else None

    for key, to_max in (("OffsetY", False), ("OffsetX", False), ("Height", True), ("Width", True)):
        name = prop(key)
        if name is None or c.isPropertyReadOnly(cam, name):
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
        if float(c.getProperty(cam, name)) != float(value):
            c.setProperty(cam, name, str(int(value)))
    c.clearROI()
    if mode is not None and mode != "FreeRunning":
        c.setProperty(cam, "TriggerMode", mode)
    print(f"sensor reset to full frame {c.getImageWidth()} x {c.getImageHeight()}")


def set_trigger_mode(c, cam, mode, real_s=0.0005):
    """A set that reaches the camera takes ~2.5 ms; one that returns in microseconds was skipped."""
    t0 = time.perf_counter()
    c.setProperty(cam, "TriggerMode", mode)
    dt = time.perf_counter() - t0
    if dt < real_s:
        print(f"TriggerMode {mode!r} set returned in {1e3 * dt:.3f} ms (camera already in it, or skipped)")


def start_software_trigger(c, cam, arm, mode="Software"):
    """Continuous sequence waiting for software pulses. arm() puts the DDS table at row 0 with
    stepping on; it is called again if untriggered frames (each of which stepped the table) arrive."""
    if c.isSequenceRunning():
        c.stopSequenceAcquisition()
    arm()
    set_trigger_mode(c, cam, "FreeRunning")  # FreeRunning, 1 s, Software, start: the sequence that works
    time.sleep(1.0)
    set_trigger_mode(c, cam, mode)
    c.clearCircularBuffer()
    c.startContinuousSequenceAcquisition(0)
    rearm = False
    if c.getProperty(cam, "TriggerMode") != mode:  # the start reset it: set it again while running
        print(f"TriggerMode reset to {c.getProperty(cam, 'TriggerMode')!r} by the start: setting {mode!r} again")
        c.setProperty(cam, "TriggerMode", mode)
        if c.getProperty(cam, "TriggerMode") != mode:
            c.stopSequenceAcquisition()
            raise SystemExit(f"TriggerMode won't stay {mode!r} once acquisition starts")
        c.clearCircularBuffer()
        rearm = True
    wait = max(0.3, 5 * c.getExposure() / 1000)  # no pulses yet, so no frames should arrive
    for attempt in range(3):
        time.sleep(wait)
        stray = c.getRemainingImageCount()
        if stray == 0:
            break
        print(f"{stray} untriggered frame(s) at start: discarding, re-arming the phase table")
        c.clearCircularBuffer()
        rearm = True
    else:
        c.stopSequenceAcquisition()
        raise SystemExit("frames keep arriving without a trigger: the camera isn't software triggering")
    if rearm:
        arm()


def software_triggered_frames(c, cam, n, pulse="TriggerSoftwarePulse"):
    """n frames, one software pulse at a time (waiting for each frame before the next pulse)."""
    frames = []
    for i in range(n):
        c.getProperty(cam, "TriggerSoftwarePulse")
        c.setProperty(cam, "TriggerSoftwarePulse", pulse)
        t_end = time.perf_counter() + 1 + 2 * c.getExposure() / 1000
        while c.getRemainingImageCount() == 0:
            if time.perf_counter() > t_end or not c.isSequenceRunning():
                raise SystemExit(f"no frame for software pulse {i + 1}/{n}")
            time.sleep(0.0005)
        frames.append(c.popNextImage())
    time.sleep(0.05)
    if c.getRemainingImageCount() > 0:
        raise SystemExit(f"{c.getRemainingImageCount()} extra untriggered frames: phases unknown")
    return frames


# ---- laser (start_laser / stop_laser in live_modulation_pmm.py)
def add_laser_args(p: argparse.ArgumentParser):
    p.add_argument("--laser-power", default="10.0", help="laser power [mW]")
    p.add_argument("--laser-power-group-name", default="640 power",
                   help="config group whose one property is the laser power")
    p.add_argument("--laser-enable-group-name", default="640 enable",
                   help="config group whose one property is the laser's digital enable (1/0)")
    p.add_argument("--no-laser", action="store_true", help="leave the laser alone")


def power_tag(args):
    """Laser power for file names: '10mW', or 'nolaser' with --no-laser."""
    return "nolaser" if args.no_laser else f"{float(args.laser_power):g}mW"


def group_property(c, group):
    """(device, property) of a config group that controls exactly one property."""
    state = c.getConfigGroupState(group)
    if state.size() != 1:
        raise SystemExit(f"Group {group} should have exactly one property! It has {state.size()}")
    setting = state.getSetting(0)
    return setting.getDeviceLabel(), setting.getPropertyName()


class Laser:
    """Power + enable from the two config groups. off() is also registered at exit, so it runs (before
    load_core's unload) even if the script dies; calling it twice is harmless."""

    def __init__(self, c, args):
        self.c, self.args = c, args
        self.power = self.enable = None

    def on(self):
        if self.args.no_laser:
            return
        c = self.c
        atexit.register(self.off)
        self.power = group_property(c, self.args.laser_power_group_name)
        c.setProperty(*self.power, self.args.laser_power)
        print(f"Laser {self.power[0]} {self.power[1]} = {c.getProperty(*self.power)}")
        self.enable = group_property(c, self.args.laser_enable_group_name)
        c.setProperty(*self.enable, "1")
        print(f"Laser {self.enable[0]} {self.enable[1]} = {c.getProperty(*self.enable)}")

    def off(self):
        """Disable, then power to 0 (or the property's lower limit); each tried even if the other fails."""
        c = self.c
        if self.enable is not None:
            try:
                c.setProperty(*self.enable, "0")
                print(f"Laser {self.enable[0]} {self.enable[1]} = {c.getProperty(*self.enable)} (off)")
                self.enable = None
            except Exception as e:
                print(f"*** laser disable failed: {e}")
        if self.power is not None:
            try:
                off = c.getPropertyLowerLimit(*self.power) if c.hasPropertyLimits(*self.power) else 0
                c.setProperty(*self.power, str(max(off, 0)))
                print(f"Laser {self.power[0]} {self.power[1]} = {c.getProperty(*self.power)} (off)")
                self.power = None
            except Exception as e:
                print(f"*** laser power to 0 failed: {e}")


# ---- 409C serial (serialFunctions.bsh)
def read_serial_reply(c, port):
    """Read lines until the 409C's status line ("OK" or "?x"), or a timeout."""
    lines = []
    while True:
        try:
            reply = c.getSerialPortAnswer(port, "\n")
        except Exception:
            break  # timeout: nothing more to read
        reply = reply.strip()  # strip any leftover \r
        if not reply:
            continue
        lines.append(reply)
        if reply == "OK" or reply.startswith("?"):
            break
    return "\n".join(lines)


def send_serial(c, port, cmd, rtn="\r\n"):
    c.setSerialPortCommand(port, cmd, rtn)


def send_serial_and_read_reply(c, port, cmd, rtn="\r\n"):
    send_serial(c, port, cmd, rtn)
    reply = read_serial_reply(c, port)
    lines = reply.split("\n")
    if lines and lines[-1].startswith("?"):
        print(f"*** 409C rejected '{cmd}': {reply}")
    return reply


def read_until_timeout(c, port, term="\r"):
    """Everything until the port times out (for commands without an OK status line)."""
    lines = []
    while True:
        try:
            lines.append(c.getSerialPortAnswer(port, term))
        except Exception:
            break
    return "\n".join(lines)
