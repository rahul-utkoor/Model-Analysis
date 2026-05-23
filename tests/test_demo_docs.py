from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_demo_readme_exists() -> None:
    assert (REPO_ROOT / "demos" / "README.md").exists()


def test_all_milestone_demo_files_exist() -> None:
    expected = [
        "milestone_01_project_setup.md",
        "milestone_02_structural_inventory.md",
        "milestone_03_dependency_graph.md",
        "milestone_04_pruning_action_simulation.md",
        "milestone_05_correspondence_shape_evidence.md",
        "milestone_06_linear_pruning_backend.md",
        "milestone_07_paired_linear_repair.md",
        "milestone_08_bert_mlp_block_pruning.md",
        "milestone_09_pruning_opportunity_map.md",
        "milestone_10_dimension_ir.md",
        "milestone_11_legality_analysis.md",
        "milestone_13_subgraph_analysis.md",
    ]
    for name in expected:
        assert (REPO_ROOT / "demos" / name).exists()


def test_full_pipeline_demo_script_exists() -> None:
    assert (REPO_ROOT / "demo_scripts" / "run_full_analysis_pipeline.sh").exists()
    assert (REPO_ROOT / "demo_scripts" / "run_demo_13_subgraph_analysis.sh").exists()


def test_glossary_mentions_dimension_variable() -> None:
    glossary = (REPO_ROOT / "demos" / "glossary.md").read_text(encoding="utf-8")
    assert "Dimension variable" in glossary


def test_readme_links_demo_track() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "demos/README.md" in readme
