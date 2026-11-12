"""
Single-molecule (peak-fitting) lifetimes for the 11 Nov data: LUT from the methylene-blue IRF
(lifetime_lut.py), then sm_lifetime.py on the sample/bead stacks.

    python run_sm_lifetime.py            # skips the LUT if it already exists
    python run_sm_lifetime.py --redo-lut
"""
import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ENV = Path(r"C:\Users\leorb\.conda\envs\image")
ROOT = Path(r"E:\20261111\20261111")
DATA = ROOT / "registered"
OUT = ROOT / "output" / "sm_lifetime"
LUT_OUT = ROOT / "output" / "lut"

IRF = DATA / "irf_tirf5048_10mW_10ms_20261111_112507_registered.tif"
SAMPLES = sorted(DATA.glob("test_*_registered.tif")) + sorted(DATA.glob("beads_*_registered.tif"))
REG = Path(r"E:\20261105\analysis\beadreg_2\bead_registration.json")  # only for <name>_registration_corrected.json

PHASE_SIGN = -1        # 22 Oct+ data only fit with -1 (lifetime_lut default is +1)
F_GATE_MHZ = 40.0      # 20 MHz RF, 180 deg gating period
PHASE_STEP = 30.0

# the env's DLL dirs must be on PATH or numpy/scipy crash silently
env = os.environ.copy()
env["PATH"] = os.pathsep.join(str(ENV / d) for d in
                              ("", r"Library\mingw-w64\bin", r"Library\usr\bin", r"Library\bin", "Scripts", "bin")) \
    + os.pathsep + env["PATH"]
py = str(ENV / "python.exe")


def run(cmd):
    print(" ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], env=env, cwd=CODE, check=True)


lut = LUT_OUT / f"{IRF.name.removesuffix('_registered.tif')}_lut.npz"
if "--redo-lut" in sys.argv or not lut.exists():
    run([py, CODE / "lifetime_lut.py", IRF, "--out", LUT_OUT, "--phase-sign", PHASE_SIGN,
         "--f-gate-mhz", F_GATE_MHZ, "--phase-step", PHASE_STEP])

run([py, CODE / "sm_lifetime.py", *SAMPLES, "--lut", lut, "--out", OUT, "--reg", REG,
     "--phase-step", PHASE_STEP])
