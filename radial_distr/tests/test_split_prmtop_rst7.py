import json
import os

import parmed as pmd
import pytest

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


def test_split_prmtop_rst7_translates_before_splitting(tmp_path):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(current_dir, "data")
    prmtop_file = os.path.join(data_dir, "2ala_2gly_dipeptides.prmtop")
    rst7_file = os.path.join(data_dir, "2ala_2gly_dipeptides.rst7")
    output_dir = str(tmp_path)
    translation = [1.5, -2.0, 3.25]

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
            "--translate",
            *[str(value) for value in translation],
        ]
    )

    translated_files = [
        "dipep_translated.prmtop",
        "dipep_translated.rst7",
        "dipep_translated.pdb",
    ]
    for filename in translated_files:
        assert os.path.isfile(os.path.join(output_dir, filename)), f"Missing generated file: {filename}"

    original = pmd.load_file(prmtop_file, rst7_file)
    translated = pmd.load_file(
        os.path.join(output_dir, "dipep_translated.prmtop"),
        os.path.join(output_dir, "dipep_translated.rst7"),
    )
    first_segment = pmd.load_file(
        os.path.join(output_dir, "dipep_001.prmtop"),
        os.path.join(output_dir, "dipep_001.rst7"),
    )

    assert translated.coordinates[0].tolist() == pytest.approx([
        original.coordinates[0][axis] + translation[axis] for axis in range(3)
    ])
    assert first_segment.coordinates[0].tolist() == pytest.approx(translated.coordinates[0].tolist())

    with open(os.path.join(output_dir, "dipep_split_summary.json")) as summary_file:
        summary = json.load(summary_file)

    assert summary["translation"] == translation
    assert summary["translated_structure"] == {
        "prmtop": os.path.join(output_dir, "dipep_translated.prmtop"),
        "rst7": os.path.join(output_dir, "dipep_translated.rst7"),
        "pdb": os.path.join(output_dir, "dipep_translated.pdb"),
    }
