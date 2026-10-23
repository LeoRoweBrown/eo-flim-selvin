"""
Acquire n_phases frames for the EO-FLIM lifetime lookup table (aquire_phases.bsh): the camera's
trigger output steps the 409C table one row per frame. Saves a (n_phases, H, W) ImageJ TIFF.

    python pymmcore_scripts/acquire_phases.py [--out lut.tif] [--exposure 100]

The table must be programmed first (set_phases_multichannel.py). Micro-Manager must be closed.
Frames are taken as in live_modulation_pmm.py's trig mode: a continuous sequence in software trigger
mode, one TriggerSoftwarePulse per frame, each frame waited for before the next pulse, and a check that
no untriggered frames (which would step the table out of sync) arrived. The .bsh's snapImage() in
free-running mode gave no such guarantee.
Unlike the .bsh (which relied on enable_phase_stepping_live.bsh having been run), this switches the
camera's trigger output on itself, and off again at the end.
The laser (--laser-power, default 10 mW) is switched on just before the first frame and off again
after the last, or on any error; --no-laser leaves it alone.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from mmcore_common import (Laser, add_common_args, add_laser_args, load_core, power_tag, reset_sensor_roi,
                           send_serial_and_read_reply, set_trigger_mode, software_triggered_frames,
                           start_software_trigger)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    add_laser_args(p)
    p.add_argument("--n-phases", type=int, default=6, help="make sure this is the same as when setting phases")
    p.add_argument("--out", type=Path, default=Path("phases.tif"),
                   help="output .tif; laser power, exposure and time are added before .tif "
                        "(default: phases_<power>mW_<exposure>ms_<time>.tif here)")
    p.add_argument("--exposure", type=float, default=None, help="camera exposure [ms] (default: as configured)")
    p.add_argument("--trigger-camera", default="TUCamGen")
    p.add_argument("--trigger-port", default="Port2")
    p.add_argument("--laser-settle-s", type=float, default=0.5, help="wait after the laser is switched on [s]")
    args = p.parse_args()
    # resolved before load_core's chdir; power, exposure and time are added at save time
    out = args.out.resolve()
    if out.suffix.lower() in (".tif", ".tiff"):
        out = out.with_suffix("")
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"

    c = load_core(args)
    port, cam = args.dds_port, args.trigger_camera

    def send(cmd):
        print(send_serial_and_read_reply(c, port, cmd))

    def arm():
        # table back to row 0, stepping with each camera trigger (TSTOP is needed before I s)
        for cmd in ("I d", "TSTOP", "TS 0", "I s"):
            send(cmd)

    laser = Laser(c, args)
    frames = []
    try:
        # same order as live_modulation_pmm.py: the Tucsen adapter is sensitive to the call sequence
        reset_sensor_roi(c, cam)  # undo a height left small by an earlier ROI (adapter bug)
        if args.exposure is not None:
            c.setExposure(args.exposure)
        c.setProperty(cam, "TriggerPort", args.trigger_port)
        c.setProperty(cam, "TriggerPortEnable", "On")
        start_software_trigger(c, cam, arm)
        laser.on()  # after the camera / DDS setup, kept out of that call sequence
        time.sleep(args.laser_settle_s)
        # one loop of phases
        frames = software_triggered_frames(c, cam, args.n_phases)
    finally:
        laser.off()  # first: the safety-relevant one
        # each step separately, so one failure doesn't leave the camera in software trigger mode
        for what, step in (("stop acquisition", lambda: c.isSequenceRunning() and c.stopSequenceAcquisition()),
                           ("DDS stepping off", lambda: [send(cmd) for cmd in ("I d", "TSTOP", "TS 0")]),
                           ("TriggerMode", lambda: set_trigger_mode(c, cam, "FreeRunning")),
                           ("trigger output off", lambda: c.setProperty(cam, "TriggerPortEnable", "Off"))):
            try:
                step()
            except Exception as e:
                print(f"*** cleanup: {what} failed: {' '.join(str(e).split())}")

    stack = np.stack(frames)
    out = out.with_name(f"{out.name}_{power_tag(args)}_{c.getExposure():g}ms_{stamp}.tif")  # lut_10mW_100ms_<time>.tif
    tifffile.imwrite(out, stack, imagej=True, metadata={"axes": "TYX"})
    print(f"saved {stack.shape} {stack.dtype} -> {out}")


if __name__ == "__main__":
    main()
