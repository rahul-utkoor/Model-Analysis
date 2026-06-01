from experimental.all_model_plan_proof.cli import main


def test_cli_help():
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
