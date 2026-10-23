"""Save every TUCamGen property (value, read-only) to a file, or compare two saved dumps.
Camera only, Micro-Manager closed. From the eoflim-pmm prompt:
    python tucsen_dump_props.py bad.txt          # now (software pulses ignored)
    ... open + close SamplePro (or whatever gives the working state) ...
    python tucsen_dump_props.py good.txt
    python tucsen_dump_props.py --diff bad.txt good.txt
"""
import ctypes
import os
import sys


def dump(out):
    import pymmcore
    di = int(pymmcore.CMMCore().getAPIVersionInfo().split("Device API version ")[1].split(",")[0])
    mm = {73: r"C:\Program Files\Micro-Manager-2.0gamma_API_73_20250530",
          74: r"C:\Program Files\Micro-Manager-2.0gamma_API_74"}[di]
    cfg = r"C:\Program Files\Micro-Manager-2.0gamma_API_73_20250530\tucsen_novatech.cfg"
    out = os.path.abspath(out)
    os.add_dll_directory(mm)
    os.environ["PATH"] = mm + os.pathsep + os.environ["PATH"]
    os.chdir(mm)
    for name in ("TUCam.dll", "mmgr_dal_Tucsen_Gen_x64.dll"):
        ctypes.WinDLL(os.path.join(mm, name))
    c = pymmcore.CMMCore()
    c.setDeviceAdapterSearchPaths([mm])
    c.loadSystemConfiguration(cfg)
    try:
        lines = []
        for name in sorted(c.getDevicePropertyNames("TUCamGen")):
            try:
                value = c.getProperty("TUCamGen", name)
            except Exception as e:
                value = f"<error {e}>"
            ro = " (read-only)" if c.isPropertyReadOnly("TUCamGen", name) else ""
            lines.append(f"{name} = {value!r}{ro}")
        open(out, "w").write("\n".join(lines) + "\n")
        print(f"{len(lines)} properties -> {out}")
    finally:
        c.unloadAllDevices()


def diff(a, b):
    def load(p):
        d = {}
        for line in open(p):
            name, _, rest = line.rstrip("\n").partition(" = ")
            d[name] = rest
        return d
    A, B = load(a), load(b)
    n = 0
    for name in sorted(set(A) | set(B)):
        if A.get(name) != B.get(name):
            n += 1
            print(f"{name}\n    {os.path.basename(a)}: {A.get(name, '<missing>')}\n    {os.path.basename(b)}: {B.get(name, '<missing>')}")
    print(f"{n} properties differ")


if __name__ == "__main__":
    if sys.argv[1] == "--diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        dump(sys.argv[1])
