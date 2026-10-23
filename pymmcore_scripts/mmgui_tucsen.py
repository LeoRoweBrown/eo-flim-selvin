"""
Start pymmcore-gui (`mmgui`) with the Tucsen camera adapter loadable.

    python pymmcore_scripts/mmgui_tucsen.py [any mmgui options]

Without this, the hardware config wizard doesn't list Tucsen_Gen_x64: MMCore finds the adapter but
can't load it, because Windows doesn't resolve the Tucsen SDK's DLL dependencies from the adapter
folder (the same problem as in live_modulation_pmm.py). Loading TUCam.dll and the adapter with ctypes
first fixes that: ctypes searches the DLL's own folder for its dependencies, and MMCore then gets the
already-loaded copies.

The adapter folder is the one pymmcore-plus uses (pymmcore_plus.find_micromanager()); set
MMGUI_MM_DIR to use another. If you point the GUI at a different Micro-Manager folder later, the
copies preloaded here are still the ones used.
"""
import ctypes
import os

from pymmcore_plus import find_micromanager

PRELOAD = ("TUCam.dll", "mmgr_dal_Tucsen_Gen_x64.dll")


def preload_tucsen(mm_dir):
    os.add_dll_directory(mm_dir)
    os.environ["PATH"] = mm_dir + os.pathsep + os.environ["PATH"]
    for name in PRELOAD:
        path = os.path.join(mm_dir, name)
        if not os.path.isfile(path):
            print(f"*** {path} not found: Tucsen camera won't be available")
            continue
        try:
            ctypes.WinDLL(path)
        except OSError as e:
            print(f"*** couldn't preload {path}: {e}")
    print(f"preloaded the Tucsen SDK + adapter from {mm_dir}")


def main():
    mm_dir = os.environ.get("MMGUI_MM_DIR") or find_micromanager()
    if mm_dir:
        preload_tucsen(str(mm_dir))
    else:
        print("*** no Micro-Manager folder found (run `mmcore install`, or set MMGUI_MM_DIR)")
    from pymmcore_gui._cli import main as mmgui_main
    mmgui_main()  # reads sys.argv itself, so mmgui's options pass through


if __name__ == "__main__":
    main()
