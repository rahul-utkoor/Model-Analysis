from experimental.bert_24_plan_proof.cli import parse_args


def test_cli_help(capsys) -> None:
    try:
        parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "Generate the BERT 24-plan" in capsys.readouterr().out
