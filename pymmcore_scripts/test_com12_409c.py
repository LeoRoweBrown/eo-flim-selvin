"""
Check the 409C DDS answers on COM12 (test_com12_409c.bsh): send "Q" and print everything it returns.

    python pymmcore_scripts/test_com12_409c.py

Micro-Manager must be closed.
"""
import argparse

from mmcore_common import add_common_args, load_core, read_until_timeout


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args()

    c = load_core(args)
    c.setSerialPortCommand(args.dds_port, "Q", "\r")
    print(read_until_timeout(c, args.dds_port, "\r"))


if __name__ == "__main__":
    main()
