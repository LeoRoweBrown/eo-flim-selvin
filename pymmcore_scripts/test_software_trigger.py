"""Python twin of test_software_trigger.bsh: same steps, same timings, via pymmcore (or pymmcore-plus).
Camera only (no DDS). Micro-Manager must be closed.

    python test_software_trigger.py                         # as live_modulation_pmm.py loads it
    python test_software_trigger.py --mm-dir "C:\\Program Files\\Micro-Manager-2.0gamma_API_74"
    python test_software_trigger.py --core plus --no-preload --no-chdir   # switch setup steps off

A TriggerMode / pulse set that reaches the camera takes ~2.5 ms (as in Micro-Manager); one that
returns in a few microseconds was only stored by the adapter.
"""
import argparse
import ctypes
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))  # before any chdir
REPO = os.path.dirname(HERE)

p = argparse.ArgumentParser()
p.add_argument("--mm-dir", default=None, help="adapter folder (default: the one matching pymmcore's device API)")
p.add_argument("--config", default=r"C:\Program Files\Micro-Manager-2.0gamma_API_73_20250530\tucsen_novatech.cfg")
p.add_argument("--core", choices=("pymmcore", "plus"), default="pymmcore")
p.add_argument("--no-preload", action="store_true", help="don't preload TUCam.dll / the adapter with ctypes")
p.add_argument("--no-chdir", action="store_true", help="don't change the working folder to --mm-dir")
p.add_argument("--read-before-pulse", action="store_true",
               help="read TriggerSoftwarePulse before each pulse (the adapter may only fire on a value change)")
g = p.add_argument_group("optional extra steps before triggering (what the failing scripts also do)")
g.add_argument("--buffer-mb", type=int, default=0, help="setCircularBufferMemoryFootprint, e.g. 2000")
g.add_argument("--exposure", type=float, default=0, help="setExposure [ms]")
g.add_argument("--read-all", action="store_true", help="read every camera property, its limits and allowed values")
g.add_argument("--clear-roi", action="store_true", help="clearROI()")
g.add_argument("--reset-roi", action="store_true", help="the live script's full-frame reset")
g.add_argument("--roi", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"), help="setROI")
g.add_argument("--trigger-out", action="store_true", help="TriggerPort=Port2, TriggerPortEnable=On")
g.add_argument("--dds", action="store_true", help="send the DDS arm commands over COM12 (I d, TSTOP, TS 0, I s)")
g.add_argument("--all", action="store_true", help="all of the above, as live_modulation_pmm.py does")
p.add_argument("--like-live", action="store_true",
               help="run live_modulation_pmm.py's startup calls in its order before the trigger test")
p.add_argument("--skip", default="",
               help="with --like-live: comma-separated steps to leave out "
                    "(buffer,init_mode,reset,sensor_mode,roi_read,trig_out,pick,dds)")
p.add_argument("--show-dlls", action="store_true",
               help="after loading, list which copy (full path) of each Tucsen-related DLL is in use")
p.add_argument("--no-pause", action="store_true", help="no 1 s pause between FreeRunning and Software")
p.add_argument("--import-first", action="store_true",
               help="import live_modulation (numpy, scipy, cv2, matplotlib, ...) before loading the camera")
p.add_argument("--leave-software", action="store_true",
               help="exit without setting TriggerMode back to FreeRunning (test: does the next session break?)")
p.add_argument("--pulses", type=int, default=3)
p.add_argument("--gap-ms", type=int, default=250)
p.add_argument("--log", default=None, help="MMCore debug log file")
a = p.parse_args()
if a.all:
    a.buffer_mb, a.read_all, a.reset_roi, a.trigger_out, a.dds = a.buffer_mb or 2000, True, True, True, True
if a.log:
    a.log = os.path.abspath(a.log)  # before any chdir

if a.import_first:  # as live_modulation_pmm.py / tucsen_trigger_test.py: analysis libraries loaded before the camera
    t0 = time.perf_counter()
    sys.path.insert(0, REPO)
    import live_modulation  # noqa: E402,F401  (numpy, scipy, cv2, matplotlib, tifffile)
    print(f"imported live_modulation and its libraries first ({time.perf_counter() - t0:.1f} s)")

import pymmcore  # noqa: E402

di = int(pymmcore.CMMCore().getAPIVersionInfo().split("Device API version ")[1].split(",")[0])
mm = a.mm_dir or {73: r"C:\Program Files\Micro-Manager-2.0gamma_API_73_20250530",
                  74: r"C:\Program Files\Micro-Manager-2.0gamma_API_74"}[di]
exe_dir = os.path.dirname(sys.executable)
print(f"python {sys.executable}; pymmcore device API {di}; adapters {mm}")
print("Tucsen files next to python.exe:",
      {f: os.path.exists(os.path.join(exe_dir, f)) for f in ("Aries 6510_PID5_SBSL16025003.xml",
                                                              "PreProcess_1.xsl", "GenApiSchema_Version_1_1.xsd")})

os.add_dll_directory(mm)
os.environ["PATH"] = mm + os.pathsep + os.environ["PATH"]
if not a.no_chdir:
    os.chdir(mm)
if not a.no_preload:
    for name in ("TUCam.dll", "mmgr_dal_Tucsen_Gen_x64.dll"):
        ctypes.WinDLL(os.path.join(mm, name))
print(f"working folder {os.getcwd()}; preload {'off' if a.no_preload else 'on'}")

if a.core == "plus":
    from pymmcore_plus import CMMCorePlus
    c = CMMCorePlus()
else:
    c = pymmcore.CMMCore()
if a.log:
    c.setPrimaryLogFile(a.log, True)
    c.enableDebugLog(True)
c.setDeviceAdapterSearchPaths([mm])
t0 = time.perf_counter()
c.loadSystemConfiguration(a.config)
print(f"config loaded in {time.perf_counter() - t0:.1f} s")


def loaded_dlls(pattern=r"tucam|tucsen|genapi|gcbase|tuimg|mvcamera|multicam|sphinx|clallserial|phxl|msvc[pr]1[24]0|vcruntime"):
    """Full paths of the DLLs this process has loaded whose name matches pattern (which copy is used)."""
    import ctypes.wintypes as wt
    import re
    psapi, k32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wt.HANDLE
    proc = k32.GetCurrentProcess()
    mods = (wt.HMODULE * 4096)()
    needed = wt.DWORD()
    psapi.EnumProcessModules.argtypes = [wt.HANDLE, ctypes.POINTER(wt.HMODULE), wt.DWORD, ctypes.POINTER(wt.DWORD)]
    psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, wt.HMODULE, wt.LPWSTR, wt.DWORD]
    psapi.EnumProcessModules(proc, mods, ctypes.sizeof(mods), ctypes.byref(needed))
    out = []
    for h in mods[:needed.value // ctypes.sizeof(wt.HMODULE)]:
        buf = ctypes.create_unicode_buffer(1024)
        psapi.GetModuleFileNameExW(proc, h, buf, 1024)
        if re.search(pattern, os.path.basename(buf.value), re.I):
            out.append(buf.value)
    return sorted(out, key=lambda s: os.path.basename(s).lower())


if a.show_dlls:
    print("loaded DLLs (which copy):")
    for path in loaded_dlls():
        print("   ", path)
cam = c.getCameraDevice()
print(f"TriggerMode at load: {c.getProperty(cam, 'TriggerMode')!r}  (left by the previous session)")


def set_timed(prop, value):
    t0 = time.perf_counter()
    c.setProperty(cam, prop, value)
    ms = 1e3 * (time.perf_counter() - t0)
    print(f"{prop} = {value}  ({ms:.3f} ms)")
    return ms


def frames_after(wait_ms):
    time.sleep(wait_ms / 1e3)
    k = c.getRemainingImageCount()
    c.clearCircularBuffer()
    return k


def dds(cmd, port="COM12"):
    c.setSerialPortCommand(port, cmd, "\r\n")
    lines = []
    while True:
        try:
            reply = c.getSerialPortAnswer(port, "\n").strip()
        except Exception:
            break
        if reply:
            lines.append(reply)
            if reply == "OK" or reply.startswith("?"):
                break
    return lines


def dds_arm():  # as live_modulation_pmm.py arm(): table to row 0, step on each camera trigger
    for cmd in ("I d", "TSTOP", "TS 0", "I s"):
        print(f"    DDS {cmd!r} -> {dds(cmd)}")


def prestep(name, fn):
    t0 = time.perf_counter()
    fn()
    print(f"pre-step {name} ({1e3 * (time.perf_counter() - t0):.1f} ms)")


def read_all():  # what the diagnostic's property dump / the full-frame reset read
    for name in c.getDevicePropertyNames(cam):
        c.getProperty(cam, name)
        c.getAllowedPropertyValues(cam, name)
        c.isPropertyReadOnly(cam, name)
        if c.hasPropertyLimits(cam, name):
            c.getPropertyLowerLimit(cam, name)
            c.getPropertyUpperLimit(cam, name)


def trigger_out():
    c.setProperty(cam, "TriggerPort", "Port2")
    c.setProperty(cam, "TriggerPortEnable", "On")


def reset_roi():
    sys.path.insert(0, REPO)
    from live_modulation_pmm import reset_sensor_roi
    reset_sensor_roi(c, cam)


try:
    print(f"camera {cam}, ROI {c.getROI()}, exposure {c.getExposure()} ms")
    if a.like_live:  # live_modulation_pmm.py's startup calls up to its trigger setup, in its order
        skip = set(a.skip.split(",")) if a.skip else set()

        def live_step(name, fn):
            if name in skip:
                print(f"live step {name}: skipped")
            else:
                prestep(f"live step {name}", fn)

        def init_mode():  # __init__: read TriggerMode, then always set FreeRunning
            c.getAllowedPropertyValues(cam, "TriggerMode")
            c.getProperty(cam, "TriggerMode")
            c.getProperty(cam, "TriggerMode")
            c.setProperty(cam, "TriggerMode", "FreeRunning")

        def sensor_mode():  # __init__: print SensorOperationMode
            if "SensorOperationMode" in c.getDevicePropertyNames(cam):
                c.getProperty(cam, "SensorOperationMode")
                c.getAllowedPropertyValues(cam, "SensorOperationMode")

        def pick():  # start_trig: choose the trigger values (reads only)
            c.isSequenceRunning()
            for prop in ("TriggerMode", "TriggerSoftwarePulse"):
                c.getAllowedPropertyValues(cam, prop)
                c.getProperty(cam, prop)

        live_step("buffer", lambda: c.setCircularBufferMemoryFootprint(2000))
        live_step("init_mode", init_mode)
        live_step("reset", reset_roi)
        live_step("sensor_mode", sensor_mode)
        live_step("roi_read", c.getROI)
        live_step("trig_out", trigger_out)
        live_step("pick", pick)
        live_step("dds", dds_arm)
    # optional extra steps the failing scripts do before triggering, in their order, to find the culprit
    if a.buffer_mb:
        prestep(f"setCircularBufferMemoryFootprint({a.buffer_mb})", lambda: c.setCircularBufferMemoryFootprint(a.buffer_mb))
    if a.exposure:
        prestep(f"setExposure({a.exposure})", lambda: c.setExposure(a.exposure))
    if a.read_all:
        prestep("read every camera property", read_all)
    if a.clear_roi:
        prestep("clearROI()", c.clearROI)
    if a.reset_roi:
        prestep("full-frame reset (reset_sensor_roi)", reset_roi)
    if a.roi:
        prestep(f"setROI{tuple(a.roi)}", lambda: c.setROI(*a.roi))
    if a.trigger_out:
        prestep("TriggerPort=Port2, TriggerPortEnable=On", trigger_out)
    if a.dds:
        prestep("DDS arm over COM12 (I d, TSTOP, TS 0, I s)", dds_arm)
    if c.isSequenceRunning():
        c.stopSequenceAcquisition()
    set_timed("TriggerMode", "FreeRunning")
    if not a.no_pause:
        time.sleep(1.0)
    set_timed("TriggerMode", "Software")

    c.clearCircularBuffer()
    c.startContinuousSequenceAcquisition(0)
    print("TriggerMode after start:", c.getProperty(cam, "TriggerMode"))
    print("untriggered frames in 500 ms:", frames_after(500))

    got = []
    for i in range(a.pulses):  # send software triggers
        before = c.getProperty(cam, "TriggerSoftwarePulse") if a.read_before_pulse else None
        if before is not None:
            print(f"  TriggerSoftwarePulse reads {before!r} before pulse {i + 1}")
        t0 = time.perf_counter()
        c.setProperty(cam, "TriggerSoftwarePulse", "TriggerSoftwarePulse")
        ms = 1e3 * (time.perf_counter() - t0)
        got.append(frames_after(a.gap_ms))
        print(f"pulse {i + 1}: set took {ms:.3f} ms, frames {got[-1]}")
    print("frames per pulse:", got, "-> OK" if got == [1] * a.pulses else "-> NOT one per pulse")
finally:
    if c.isSequenceRunning():
        c.stopSequenceAcquisition()
    if a.leave_software:
        print("leaving the camera in TriggerMode 'Software' (--leave-software)")
    else:
        set_timed("TriggerMode", "FreeRunning")
    c.clearCircularBuffer()
    c.unloadAllDevices()
