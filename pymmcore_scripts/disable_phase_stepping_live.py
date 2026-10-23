"""
Disable free-running phase stepping (disable_phase_stepping_live.bsh): DDS table stepping off,
table back to row 0, camera trigger output off.

    python pymmcore_scripts/disable_phase_stepping_live.py

Micro-Manager must be closed.
"""
import argparse

from mmcore_common import add_common_args, load_core, send_serial_and_read_reply


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--trigger-camera", default="TUCamGen")
    p.add_argument("--trigger-port", default="Port2")
    args = p.parse_args()

    c = load_core(args)
    port = args.dds_port

    def send(cmd):
        print(send_serial_and_read_reply(c, port, cmd))

    send("I d")  # disable table stepping with camera trigger
    send("TSTOP")  # stop table running (not sure if necessary)
    send("TS 0")  # set the table row to the beginning

    c.setProperty(args.trigger_camera, "TriggerPort", args.trigger_port)
    c.setProperty(args.trigger_camera, "TriggerPortEnable", "Off")


if __name__ == "__main__":
    main()
