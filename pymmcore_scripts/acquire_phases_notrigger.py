"""
Acquire n_phases frames for the EO-FLIM lifetime lookup table, setting each 409C table row over
serial before its snap instead of camera-triggered stepping (acquire_phases_notrigger.bsh).
Saves a (n_phases, H, W) ImageJ TIFF.

    python pymmcore_scripts/acquire_phases_notrigger.py [--out lut.tif] [--exposure 100]

The table must be programmed first (set_phases_multichannel.py). Micro-Manager must be closed.
The laser (--laser-power, default 10 mW) is switched on just before the first frame and off again
after the last, or on any error; --no-laser leaves it alone.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from mmcore_common import Laser, add_common_args, add_laser_args, load_core, power_tag, send_serial_and_read_reply


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    add_laser_args(p)
    p.add_argument("--n-phases", type=int, default=6, help="make sure this is the same as when setting phases")
    p.add_argument("--out", type=Path, default=Path("phases.tif"),
                   help="output .tif; laser power, exposure and time are added before .tif "
                        "(default: phases_<power>mW_<exposure>ms_<time>.tif here)")
    p.add_argument("--exposure", type=float, default=None, help="camera exposure [ms] (default: as configured)")
    args = p.parse_args()
    # resolved before load_core's chdir; power, exposure and time are added at save time
    out = args.out.resolve()
    if out.suffix.lower() in (".tif", ".tiff"):
        out = out.with_suffix("")
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"

    c = load_core(args)
    port = args.dds_port

    def send(cmd):
        print(send_serial_and_read_reply(c, port, cmd))

    if args.exposure is not None:
        c.setExposure(args.exposure)
    send("TS 0")  # set the table row to the beginning

    laser = Laser(c, args)
    frames = []
    try:
        laser.on()
        # one loop of phases
        for n_p in range(args.n_phases):
            send(f"TS {n_p}")  # set row
            c.snapImage()
            frames.append(c.getImage())
            time.sleep(0.01)  # sleep for safety
    finally:
        laser.off()  # first: the safety-relevant one
        send("TS 0")
        send("TSTOP")

    stack = np.stack(frames)
    out = out.with_name(f"{out.name}_{power_tag(args)}_{c.getExposure():g}ms_{stamp}.tif")  # lut_10mW_100ms_<time>.tif
    tifffile.imwrite(out, stack, imagej=True, metadata={"axes": "TYX"})
    print(f"saved {stack.shape} {stack.dtype} -> {out}")


if __name__ == "__main__":
    main()
