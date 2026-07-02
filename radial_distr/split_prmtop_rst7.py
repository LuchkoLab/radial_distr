#!/usr/bin/env python3
"""
Split an AMBER prmtop + rst7 into one prmtop + rst7 per bonded (connected) molecule.

Usage:
  python3 radial_distr/split_prmtop_rst7.py -p in.prmtop -r in.rst7 -o outdir --prefix prefix
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def load_parmed():
    """Import ParmEd when the split workflow actually needs it."""
    try:
        import parmed as pmd
    except Exception:
        print("Error: ParmEd is required (pip install parmed).", file=sys.stderr)
        raise
    return pmd


def find_components(struct: pmd.Structure):
    """Return atom-index groups for each bonded component in a structure.

    ParmEd stores bonds as atom-object pairs, so this builds an adjacency list
    over zero-based atom indices and walks it with depth-first search. Atoms
    with no bonds are returned as single-atom components.
    """
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


def translate_structure(struct: pmd.Structure, translation: tuple[float, float, float]) -> None:
    """Translate all coordinates in-place by an x/y/z vector."""
    if struct.coordinates is None:
        raise ValueError("Input structure does not contain coordinates to translate")

    dx, dy, dz = translation
    struct.coordinates = struct.coordinates + [dx, dy, dz]


def save_structure(struct: pmd.Structure, out_prmtop: str, out_rst7: str, out_pdb: str) -> None:
    """Save a full structure as AMBER topology, restart, and PDB files."""
    struct.save(out_prmtop, overwrite=True)
    struct.save(out_rst7, overwrite=True)
    struct.save(out_pdb, overwrite=True)


def save_component(struct: pmd.Structure, indices: list[int], out_prmtop: str, out_rst7: str, out_pdb: str):
    """Save a selected component as AMBER topology, restart, and PDB files.

    Args:
        struct: ParmEd structure containing the full system and coordinates.
        indices: Zero-based atom indices to extract as one component.
        out_prmtop: Destination path for the component topology.
        out_rst7: Destination path for the component coordinates.
        out_pdb: Destination path for a PDB copy of the component.

    Raises:
        RuntimeError: If the installed ParmEd version cannot slice a structure
            with an integer index list.
    """
    try:
        sub = struct[indices]
    except Exception as e:
        raise RuntimeError(
            "ParmEd selection by integer-index list failed; upgrade ParmEd or report the error"
        ) from e
    sub.save(out_prmtop, overwrite=True)
    sub.save(out_rst7, overwrite=True)
    sub.save(out_pdb, overwrite=True)


def build_parser():
    """Build the command-line parser for splitting AMBER structure files."""
    p = argparse.ArgumentParser(description="Split AMBER prmtop + rst7 by bonded molecules.")
    p.add_argument("--prmtop", "-p", required=True, help="Input prmtop/parm7 file")
    p.add_argument("--rst7", "-r", required=True, help="Input rst7 file (coordinates)")
    p.add_argument("--outdir", "-o", default="split_out", help="Output directory")
    p.add_argument("--prefix", default="out", help="Output filename prefix")
    p.add_argument("--pad", type=int, default=3, help="Zero-pad width for numeric index")
    p.add_argument(
        "--translate",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Translate all input coordinates by this x/y/z vector before saving and splitting",
    )
    p.add_argument(
        "--translated-prefix",
        help="Output filename prefix for the full translated structure (default: <prefix>_translated)",
    )
    p.add_argument(
        "--no-save-translated",
        action="store_true",
        help="Translate before splitting without saving the full translated structure",
    )

    return p


def main(argv=None):
    """Run the split workflow from command-line-style arguments.

    Args:
        argv: Optional argument list. When omitted, argparse reads arguments
            from ``sys.argv``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    pmd = load_parmed()
    struct = pmd.load_file(args.prmtop, args.rst7)
    translation = tuple(args.translate) if args.translate else None
    translated_outputs = None

    if translation is not None:
        translate_structure(struct, translation)
        if not args.no_save_translated:
            translated_prefix = args.translated_prefix or f"{args.prefix}_translated"
            out_prmtop = os.path.join(args.outdir, f"{translated_prefix}.prmtop")
            out_rst7 = os.path.join(args.outdir, f"{translated_prefix}.rst7")
            out_pdb = os.path.join(args.outdir, f"{translated_prefix}.pdb")
            save_structure(struct, out_prmtop, out_rst7, out_pdb)
            translated_outputs = {"prmtop": out_prmtop, "rst7": out_rst7, "pdb": out_pdb}
            print(
                f"Wrote translated structure {out_prmtop} and {out_rst7} and {out_pdb} "
                f"(translation: {translation[0]} {translation[1]} {translation[2]})"
            )

    comps = find_components(struct)

    written = 0
    summary = []
    for idx, comp in enumerate(comps, start=1):
        name = f"{args.prefix}_{idx:0{args.pad}d}"
        out_prmtop = os.path.join(args.outdir, f"{name}.prmtop")
        out_rst7 = os.path.join(args.outdir, f"{name}.rst7")
        out_pdb = os.path.join(args.outdir, f"{name}.pdb")
        save_component(struct, comp, out_prmtop, out_rst7, out_pdb)
        written += 1
        print(f"Wrote {out_prmtop} ({len(comp)} atoms) and {out_rst7} and {out_pdb}")
        summary.append(
            {
                "index": idx,
                "name": name,
                "n_atoms": len(comp),
                "prmtop": out_prmtop,
                "rst7": out_rst7,
                "pdb": out_pdb,
            }
        )

    # write summary
    summary_path = os.path.join(args.outdir, f"{args.prefix}_split_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "n_components": len(comps),
                "written": written,
                "translation": list(translation) if translation is not None else None,
                "translated_structure": translated_outputs,
                "components": summary,
            },
            fh,
            indent=2,
        )

    print(f"Done: {len(comps)} components found, {written} files written to {args.outdir}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
