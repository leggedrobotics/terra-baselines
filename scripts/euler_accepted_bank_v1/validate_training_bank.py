#!/usr/bin/env python3
"""Validate the complete accepted-bank training payload before upload."""

import argparse

from utils.accepted_bank import validate_staged_training_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank_root")
    args = parser.parse_args()
    print(validate_staged_training_bank(args.bank_root))


if __name__ == "__main__":
    main()
