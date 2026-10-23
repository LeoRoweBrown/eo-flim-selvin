"""
Build a 409C phase-stepping table with per-channel settings (set_phases_multichannel.bsh).

    python pymmcore_scripts/set_phases_multichannel.py

Edit the settings below. Micro-Manager must be closed.
"""
import argparse

from mmcore_common import add_common_args, load_core, send_serial_and_read_reply

N_PHASES = 6
DWELL_US = 1000
PHASE_RANGE_DEG = 180.0  # phases span 0 .. range*(n-1)/n

# Per-channel settings, index = 409C output channel 0..3
ENABLED = [True, True, False, False]
FREQ_MHZ = [20.0, 20.0, 20.0, 20.0]
VOLTAGE_PC = [0.122, 0.122, 0.1, 0.1]  # Vpp, 0 to 1.000
OFFSET_DEG = [0.0, 0.0, 0.0, 0.0]  # fixed phase offset
STEP_PHASE = [True, True, False, False]  # step this channel's phase?


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args()
    program_table(args, VOLTAGE_PC)


def program_table(args, voltage_pc, save=True):
    """Program and check the table with these per-channel voltages (rf_off.py passes zeros)."""
    n_step = PHASE_RANGE_DEG / N_PHASES
    # check the minimum dwell (13/19/25/31 us for 1-4 channels)
    n_ch = sum(ENABLED)
    if n_ch == 0:
        raise SystemExit("No channels enabled")
    min_dwell = 13 + 6 * (n_ch - 1)
    if DWELL_US < min_dwell:
        raise SystemExit(f"DWELL_US must be >= {min_dwell} us for {n_ch} channels")

    print("Starting set_phases_multichannel")
    c = load_core(args)
    port = args.dds_port

    def send(cmd):
        print(send_serial_and_read_reply(c, port, cmd))

    print("Setting up table")
    send("E d")
    send("TSTOP")  # stop before clearing
    send("TCLEAR")

    phases_deg = [[0.0] * 4 for _ in range(N_PHASES)]
    for n_p in range(N_PHASES):
        # row format: T row dwell [ch freq phase amp] x up to 4
        line = f"T {n_p} {DWELL_US}"
        for ch in range(4):
            if not ENABLED[ch]:
                continue
            ph = OFFSET_DEG[ch] + (n_step * n_p if STEP_PHASE[ch] else 0.0)
            ph = (round(ph * 100.0) / 100.0) % 360.0  # 2 decimals, 0..359.99
            phases_deg[n_p][ch] = ph
            line += f" {ch} {FREQ_MHZ[ch]:.7f} {ph:.2f} {voltage_pc[ch]:.3f}"
        print(f"line to set serial: {line}")
        send(line)

    if save:
        send("TSAVE")
    send(f"TRNG 0 {N_PHASES - 1}")
    # leave the table stopped at row 0, as disable_phase_stepping_live: "I d" only turns off stepping
    # on the camera trigger; a table already running on its own dwell timer needs TSTOP
    send("I d")
    send("TSTOP")
    send("TS 0")
    # check table is correct
    send(f"D 0 {N_PHASES - 1}")

    # summary of programmed phases
    for n_p in range(N_PHASES):
        print(f"row {n_p}:" + "".join(f"  ch{ch}={phases_deg[n_p][ch]} deg" for ch in range(4) if ENABLED[ch]))


if __name__ == "__main__":
    main()
