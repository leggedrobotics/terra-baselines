#!/usr/bin/env python3
"""Render a bank condition's layouts (target/occupancy/dumpability) to a PNG.

Usage:
    python scripts/view_bank_condition.py trn-net4-side1-road
    python scripts/view_bank_condition.py fnd-slab-apron-d16 --n 8 --out /tmp/d16.png
    python scripts/view_bank_condition.py --list            # show all condition names

Target colormap: blue = dig cells (negative), red = accepted dump mask
(positive). Occupancy shows obstacles plus road strips; dumpability shows
where dumping is allowed.
"""

import argparse
import glob
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_BANK = (
    "/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2/train"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", nargs="?", help="condition name (suffix match)")
    parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--n", type=int, default=4, help="layouts to render")
    parser.add_argument("--out", default=None, help="output PNG (default: /tmp/<condition>.png)")
    parser.add_argument("--list", action="store_true", help="list conditions and exit")
    parser.add_argument("--no-open", action="store_true", help="do not xdg-open the PNG")
    args = parser.parse_args()

    if args.list or not args.condition:
        for d in sorted(glob.glob(args.bank + "/*__*")):
            print(os.path.basename(d).split("__", 1)[1])
        return

    matches = sorted(glob.glob(f"{args.bank}/*__*{args.condition}*"))
    if not matches:
        raise SystemExit(f"no condition matching {args.condition!r} under {args.bank}")
    d = matches[0]
    name = os.path.basename(d).split("__", 1)[1]
    files = sorted(glob.glob(d + "/images/*.npy"))[: args.n]

    fig, axes = plt.subplots(3, len(files), figsize=(3.5 * len(files), 10), squeeze=False)
    for i, f in enumerate(files):
        img = np.load(f)
        occ = np.load(f.replace("/images/", "/occupancy/"))
        dmp = np.load(f.replace("/images/", "/dumpability/"))
        axes[0][i].imshow(img, cmap="coolwarm")
        axes[0][i].set_title(f"{os.path.basename(f)}: target")
        axes[1][i].imshow(occ, cmap="gray_r")
        axes[1][i].set_title("occupancy")
        axes[2][i].imshow(dmp, cmap="viridis")
        axes[2][i].set_title("dumpability")
    for row in axes:
        for ax in row:
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(name)
    plt.tight_layout()
    out = args.out or f"/tmp/{name}.png"
    plt.savefig(out, dpi=110)
    print(out)
    if not args.no_open:
        subprocess.Popen(["xdg-open", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
