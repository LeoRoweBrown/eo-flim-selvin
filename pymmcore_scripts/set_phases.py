"""
Build a single-channel 409C phase-stepping table (set_phases.bsh). set_phases_multichannel.py is the
newer, more general version.

    python pymmcore_scripts/set_phases.py

Edit the settings below. Micro-Manager must be closed.
"""
import argparse
import time

from mmcore_common import add_common_args, load_core, send_serial_and_read_reply

DDS_OUTPUT_CHANNEL = 0
DDS_OUTPUT_FREQUENCY = 20.0  # MHz
VOLTAGE_PC = 0.1
N_PHASES = 6
DWELL_US = 1000


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args()

    print("Starting set_phases")
    c = load_core(args)
    port = args.dds_port
    n_step = 180.0 / N_PHASES

    def send(cmd):
        reply = send_serial_and_read_reply(c, port, cmd)
        print(reply)
        return reply

    print("Setting up table")
    send("TCLEAR")
    send("TSTOP")
    send(f"TRNG 0 {N_PHASES - 1}")

    for n_p in range(N_PHASES):
        phase = n_step * n_p
        # line format: row, dwell, channel, frequency, phase, amplitude
        line = f"T {n_p} {DWELL_US} {DDS_OUTPUT_CHANNEL} {DDS_OUTPUT_FREQUENCY:.2f} {phase:.2f} {VOLTAGE_PC:.2f}"
        print(f"line to set serial: {line}")
        send(line)
        time.sleep(0.1)  # safety sleep

    send("TSAVE")

    # check table is correct
    send(f"D 0 {N_PHASES - 1}")


if __name__ == "__main__":
    main()
