import json
import os

import parmed as pmd

import radial_distr.split_prmtop_rst7 as split_prmtop_rst7


KEEP_SPLIT_OUTPUTS = False  # Set to True to keep the split outputs for inspection


def test_split_prmtop_rst7_writes_expected_components(tmp_path):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(current_dir, "data")
    prmtop_file = os.path.join(data_dir, "2ala_2gly_dipeptides.prmtop")
    rst7_file = os.path.join(data_dir, "2ala_2gly_dipeptides.rst7")
    output_dir = os.path.join(current_dir, "split_prmtop_rst7_output") if KEEP_SPLIT_OUTPUTS else str(tmp_path)
    os.makedirs(output_dir, exist_ok=True)

    split_prmtop_rst7.main(
        [
            "--prmtop",
            prmtop_file,
            "--rst7",
            rst7_file,
            "--outdir",
            output_dir,
            "--prefix",
            "dipep",
        ]
    )

    expected_atom_counts = [22, 22, 19, 19]
    expected_files = [
        "dipep_split_summary.json",
        *[f"dipep_{index:03d}.prmtop" for index in range(1, 5)],
        *[f"dipep_{index:03d}.rst7" for index in range(1, 5)],
    ]

    for filename in expected_files:
        assert os.path.isfile(os.path.join(output_dir, filename)), f"Missing generated file: {filename}"

    with open(os.path.join(output_dir, "dipep_split_summary.json")) as summary_file:
        summary = json.load(summary_file)

    assert summary["n_components"] == 4
    assert summary["written"] == 4
    assert [component["n_atoms"] for component in summary["components"]] == expected_atom_counts

    roundtrip_atom_counts = []
    roundtrip_residue_counts = []
    roundtrip_coordinate_counts = []
    for index in range(1, 5):
        component = pmd.load_file(
            os.path.join(output_dir, f"dipep_{index:03d}.prmtop"),
            os.path.join(output_dir, f"dipep_{index:03d}.rst7"),
        )
        roundtrip_atom_counts.append(len(component.atoms))
        roundtrip_residue_counts.append(len(component.residues))
        roundtrip_coordinate_counts.append(len(component.coordinates))

    assert roundtrip_atom_counts == expected_atom_counts
    assert roundtrip_residue_counts == [3, 3, 3, 3]
    assert roundtrip_coordinate_counts == expected_atom_counts
