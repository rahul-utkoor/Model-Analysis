from scripts.build_static_pruning_formalization import parse_args


def test_cli_help(capsys) -> None:
    try:
        parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "formalization" in capsys.readouterr().out.lower()
