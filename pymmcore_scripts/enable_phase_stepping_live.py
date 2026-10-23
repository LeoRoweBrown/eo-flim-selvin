"""
Enable free-running phase stepping (enable_phase_stepping_live.bsh): the camera's trigger output
steps the 409C table one row per frame.

    python pymmcore_scripts/enable_phase_stepping_live.py

Micro-Manager must be closed. In the GUI this was run before live mode in the same session; as a
standalone script it sets the DDS (which keeps its state) and the camera's trigger output, then
releases the hardware. live_modulation_pmm.py and acquire_phases.py do this themselves.
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

    send("TSTOP")  # stop table, needed before I s
    send("TS 0")  # set the table row to the beginning

    c.setProperty(args.trigger_camera, "TriggerPort", args.trigger_port)
    c.setProperty(args.trigger_camera, "TriggerPortEnable", "On")

    send("I s")  # now start: enable table stepping with camera trigger


if __name__ == "__main__":
    main()
