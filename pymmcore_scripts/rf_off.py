"""
Turn the Pockels cell RF drive off: reprogram the 409C table from set_phases_multichannel.py with
every channel's voltage at 0 (same frequencies and phases), stopped at row 0.

    python pymmcore_scripts/rf_off.py          # RF off
    python pymmcore_scripts/set_phases_multichannel.py   # RF back on (the real table)

Writes the same row format as set_phases_multichannel, so it doesn't depend on any other 409C
command. The zero table is not TSAVEd by default, so the real table stays in the 409C's flash
(--save to store the zero table too). Check on a scope / RF power meter that the drive really is off.
The amplifier still idles (and warms) with no RF in: switch it off too for a full cool-down test.
Micro-Manager must be closed.
"""
import argparse

from mmcore_common import add_common_args
from set_phases_multichannel import program_table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--save", action="store_true", help="TSAVE the zero table (default: RAM only)")
    args = p.parse_args()
    program_table(args, [0.0] * 4, save=args.save)
    print("RF voltage 0 on all channels")


if __name__ == "__main__":
    main()
