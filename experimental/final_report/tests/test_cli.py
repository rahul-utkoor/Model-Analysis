from experimental.final_report.cli import parse_args


def test_cli_help(capsys) -> None:
    try:
        parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "final static pruning propagation" in capsys.readouterr().out.lower()
