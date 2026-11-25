"""v0.5 WAR 명령 (cli_war) — 등록·안전장치·war.json 쓰기."""

import json
from pathlib import Path

import pytest

from mlbproj import cli, cli_war
from mlbproj import contracts as c
from mlbproj.models.params import NO_ADOPTION, Adoption

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "tests" / "fixtures" / "synthetic" / "web" / "war.json"


def test_cli_war_commands_registered() -> None:
    parser = cli.build_parser()
    for name in cli_war.WAR_COMMANDS:
        extra = ["--target", "2026"] if name == "final-test-war" else []
        args = parser.parse_args([name, *extra])
        assert args.command == name


def test_cli_war_final_test_accepts_only_2026() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["final-test-war", "--target", "2025"])


@pytest.mark.parametrize("metric", ["pa", "ip"])
def test_cli_war_playing_time_guard_rejects_adopted_volume(metric: str) -> None:
    adopt = dict(NO_ADOPTION.adopt)
    adopt[metric] = True
    with pytest.raises(ValueError, match="출전량 보정"):
        cli_war.require_playing_time_unadopted(Adoption(version="x", adopt=adopt))


def test_cli_war_playing_time_guard_accepts_current_adoption() -> None:
    cli_war.require_playing_time_unadopted(NO_ADOPTION)
    committed = ROOT / c.ADOPTION_PATH
    if committed.exists():  # 커밋된 v1 채택표는 PA·IP를 끈다(E23)
        from mlbproj.models.params import load_adoption

        cli_war.require_playing_time_unadopted(load_adoption(committed))


def test_cli_war_write_war_json_keeps_player_files(tmp_path: Path) -> None:
    players = tmp_path / "players"
    players.mkdir()
    keep = players / "660271.json"
    keep.write_text("{}", encoding="utf-8")
    other = tmp_path / "batters.json"
    other.write_text("{}", encoding="utf-8")
    content = json.loads(SAMPLE.read_text(encoding="utf-8"))
    size = cli_war.write_war_json(content, tmp_path)
    assert size == (tmp_path / "war.json").stat().st_size > 0
    assert keep.read_text(encoding="utf-8") == "{}"  # write_web과 달리 선수 상세를 지우지 않는다
    assert other.read_text(encoding="utf-8") == "{}"


def test_cli_war_write_war_json_rejects_schema_violation(tmp_path: Path) -> None:
    content = json.loads(SAMPLE.read_text(encoding="utf-8"))
    content["rows"][0]["roles"] = "ghost"
    with pytest.raises(ValueError, match="WarFile"):
        cli_war.write_war_json(content, tmp_path)
    assert not (tmp_path / "war.json").exists()


def test_cli_war_tune_rejects_final_target_before_reading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "tune-war",
            "--targets",
            "2024,2026",
            "--interim-dir",
            str(tmp_path / "nope"),
            "--raw-dir",
            str(tmp_path / "nope"),
            "--out-dir",
            str(tmp_path / "out"),
            "--war-params-out",
            str(tmp_path / "war_v1.json"),
        ]
    )
    assert code == 1
    assert not (tmp_path / "war_v1.json").exists()
    assert "중단" in capsys.readouterr().err
