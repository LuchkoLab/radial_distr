#!/usr/bin/env python3
"""
Split an AMBER prmtop + rst7 into one prmtop + rst7 per bonded (connected) molecule.

Usage:
  python3 radial_distr/split_prmtop_rst7.py -p in.prmtop -r in.rst7 -o outdir -prefix prefix
"""
from __future__ import annotations

import argparse
import os
import sys
import json

try:
    import parmed as pmd
except Exception:
    print("Error: ParmEd is required (pip install parmed).", file=sys.stderr)
    raise


def find_components(struct: pmd.Structure):
    natoms = len(struct.atoms)
    atom_index = {atom: i for i, atom in enumerate(struct.atoms)}
    adj = [[] for _ in range(natoms)]
    for bond in struct.bonds:
        i = atom_index[bond.atom1]
        j = atom_index[bond.atom2]
        adj[i].append(j)
        adj[j].append(i)
    visited = [False] * natoms
    comps: list[list[int]] = []
    for i in range(natoms):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def save_component(struct: pmd.Structure, indices: list[int], out_prmtop: str, out_rst7: str):
    try:
        sub = struct[indices]
    except Exception as e:
        raise RuntimeError(
            "ParmEd selection by integer-index list failed; upgrade ParmEd or report the error"
        ) from e
    sub.save(out_prmtop)
    sub.save(out_rst7)


def build_parser():
    p = argparse.ArgumentParser(description="Split AMBER prmtop + rst7 by bonded molecules.")
    p.add_argument("--prmtop", "-p", required=True, help="Input prmtop/parm7 file")
    p.add_argument("--rst7", "-r", required=True, help="Input rst7 file (coordinates)")
    p.add_argument("--outdir", "-o", default="split_out", help="Output directory")
    p.add_argument("--prefix", default="out", help="Output filename prefix")
    p.add_argument("--pad", type=int, default=3, help="Zero-pad width for numeric index")
    
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    struct = pmd.load_file(args.prmtop, args.rst7)
    comps = find_components(struct)

    written = 0
    summary = []
    for idx, comp in enumerate(comps, start=1):
        
        name = f"{args.prefix}_{idx:0{args.pad}d}"
        out_prmtop = os.path.join(args.outdir, f"{name}.prmtop")
        out_rst7 = os.path.join(args.outdir, f"{name}.rst7")
        save_component(struct, comp, out_prmtop, out_rst7)
        written += 1
        print(f"Wrote {out_prmtop} ({len(comp)} atoms) and {out_rst7}")
        summary.append({"index": idx, "name": name, "n_atoms": len(comp), "prmtop": out_prmtop, "rst7": out_rst7})

    # write summary
    summary_path = os.path.join(args.outdir, f"{args.prefix}_split_summary.json")
    with open(summary_path, "w") as fh:
        json.dump({"n_components": len(comps), "written": written, "components": summary}, fh, indent=2)

    print(f"Done: {len(comps)} components found, {written} files written to {args.outdir}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
