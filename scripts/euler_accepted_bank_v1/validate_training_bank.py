#!/usr/bin/env python3
"""Validate the complete accepted-bank training payload before upload."""

import argparse

from utils.accepted_bank import validate_staged_training_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank_root")
    parser.add_argument("--expected-maps-per-condition", type=int)
    parser.add_argument("--expected-release-id")
    args = parser.parse_args()
    print(
        validate_staged_training_bank(
            args.bank_root,
            expected_maps_per_condition=args.expected_maps_per_condition,
            expected_release_id=args.expected_release_id,
        )
    )


if __name__ == "__main__":
    main()
