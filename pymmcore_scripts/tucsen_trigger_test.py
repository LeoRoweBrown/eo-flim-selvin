"""Which ROI / software-trigger combinations make the Tucsen produce one frame per pulse?
Camera only (no DDS commands). Micro-Manager must be closed. Run from the eoflim prompt:
    python tucsen_trigger_test.py
"""
import ctypes
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))  # before any chdir
REPO = os.path.dirname(HERE)

import sys

import pymmcore

sys.path.insert(0, REPO)
from live_modulation_pmm import reset_sensor_roi  # noqa: E402

# adapters matching this pymmcore's device interface (73 in eoflim-pmm, 74 in eoflim)
DI = int(pymmcore.CMMCore().getAPIVersionInfo().split("Device API version ")[1].split(",")[0])
MM = {73: r"C:\Program Files\Micro-Manager-2.0gamma_API_73_20250530",
      74: r"C:\Program Files\Micro-Manager-2.0gamma_API_74"}[DI]
print(f"device interface {DI}: adapters from {MM}")
CFG = r"C:\Program Files\Micro-Manager-2.0gamma_API_73_20250530\tucsen_novatech.cfg"
CAM = "TUCamGen"
ROI = (400, 400, 2400, 2400)   
EXP_MS = 10

os.add_dll_directory(MM)
os.environ["PATH"] = MM + os.pathsep + os.environ["PATH"]
os.chdir(MM)
for name in ("TUCam.dll", "mmgr_dal_Tucsen_Gen_x64.dll"):
    ctypes.WinDLL(os.path.join(MM, name))
c = pymmcore.CMMCore()
LOG = os.path.join(HERE, "tucsen_mm.log")
c.setPrimaryLogFile(LOG, True)   # MMCore + Tucsen adapter messages, incl. its SDK calls
c.enableDebugLog(True)
print(f"MMCore debug log: {LOG}")
c.setDeviceAdapterSearchPaths([MM])
c.loadSystemConfiguration(CFG)
c.setExposure(EXP_MS)


def dump_roi_props(label):
    """Every camera property that could hold the ROI / sensor size, with value and limits."""
    print(f"--- {label}: getROI {c.getROI()}, image {c.getImageWidth()} x {c.getImageHeight()}")
    for name in c.getDevicePropertyNames(CAM):
        if any(k in name.lower() for k in ("roi", "height", "width", "offset", "bin", "sensor", "region", "area")):
            lim = ""
            if c.hasPropertyLimits(CAM, name):
                lim = f"  limits {c.getPropertyLowerLimit(CAM, name)}..{c.getPropertyUpperLimit(CAM, name)}"
            allowed = list(c.getAllowedPropertyValues(CAM, name))
            ro = "  (read-only)" if c.isPropertyReadOnly(CAM, name) else ""
            print(f"    {name!r} = {c.getProperty(CAM, name)!r}{lim}"
                  + (f"  allowed {allowed[:8]}" if allowed else "") + ro)


dump_roi_props("after loading the config")
reset_sensor_roi(c, CAM)
dump_roi_props("after the full-frame reset")


def stop():
    if c.isSequenceRunning():
        c.stopSequenceAcquisition()


def frames_after(wait_s):
    time.sleep(wait_s)
    k = c.getRemainingImageCount()
    c.clearCircularBuffer()
    return k


def pulses(k=3, gap_s=0.25):
    got = []
    for _ in range(k):
        c.setProperty(CAM, "TriggerSoftwarePulse", "TriggerSoftwarePulse")
        got.append(frames_after(gap_s))
    return got


def trigger_out(on):
    """The camera's trigger output, as live_modulation_pmm.py sets it for the DDS."""
    if on:
        c.setProperty(CAM, "TriggerPort", "Port2")
    c.setProperty(CAM, "TriggerPortEnable", "On" if on else "Off")


def case(name, roi, snap_first, apply_roi_by_freerun, set_mode_after_start, trig_out=None):
    """trig_out: None (output off), "before" / "after" the ROI is set, "after_mode" (after
    TriggerMode = Software, before starting) or "after_start"."""
    stop()
    c.setProperty(CAM, "TriggerMode", "FreeRunning")
    trigger_out(trig_out == "before")
    if roi:
        c.setROI(*roi)
    else:
        reset_sensor_roi(c, CAM)
    if trig_out == "after":
        trigger_out(True)
    if snap_first:
        c.snapImage()
        c.getImage()
    if apply_roi_by_freerun:  # start + stop once in FreeRunning so the ROI is applied before triggering
        c.startContinuousSequenceAcquisition(0)
        time.sleep(0.3)
        stop()
        c.clearCircularBuffer()
    if not set_mode_after_start:
        c.setProperty(CAM, "TriggerMode", "Software")
    if trig_out == "after_mode":
        trigger_out(True)
    c.clearCircularBuffer()
    c.startContinuousSequenceAcquisition(0)
    if trig_out == "after_start":
        trigger_out(True)
    if set_mode_after_start:
        try:
            c.setProperty(CAM, "TriggerMode", "Software")
        except Exception as e:
            print(f"{name}: can't set TriggerMode while running ({' '.join(str(e).split())[:120]})")
            stop()
            return
    mode = c.getProperty(CAM, "TriggerMode")
    stray = frames_after(0.4)
    got = pulses()
    x, y, w, h = c.getROI()
    stop()
    ok = "OK" if got == [1, 1, 1] and stray == 0 else "--"
    print(f"{ok} {name:45s} ROI {w}x{h}@({x},{y})  mode after start {mode!r:13s} "
          f"untriggered {stray:3d}  frames per pulse {got}")


try:
    case("full frame", None, False, False, False)
    case("ROI", ROI, False, False, False)
    case("ROI + snap first", ROI, True, False, False)
    case("ROI applied by a FreeRunning start/stop", ROI, False, True, False)
    case("ROI, Software set after starting", ROI, False, False, True)
    case("full frame again", None, False, False, False)
    case("full frame + trigger out", None, False, False, False, "after")
    case("ROI + trigger out set after the ROI", ROI, False, False, False, "after")
    case("ROI + snap + trigger out after (= script)", ROI, True, False, False, "after")
    case("ROI + trigger out set before the ROI", ROI, False, False, False, "before")
    case("ROI + trigger out before, snap", ROI, True, False, False, "before")
    case("ROI + trigger out after TriggerMode", ROI, True, False, False, "after_mode")
    case("ROI + trigger out after starting", ROI, True, False, False, "after_start")
    case("full frame + trigger out after starting", None, False, False, False, "after_start")
    trigger_out(False)
    dump_roi_props("at the end")
finally:
    stop()
    c.setProperty(CAM, "TriggerMode", "FreeRunning")
    reset_sensor_roi(c, CAM)
    c.unloadAllDevices()
