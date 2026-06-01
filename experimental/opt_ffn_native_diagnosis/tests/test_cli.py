from experimental.opt_ffn_native_diagnosis.cli import parse_args


def test_cli_help(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["opt-ffn-native-diagnosis", "--help"])
    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code == 0
