"""build 명령 테스트 — 검증 통과 확인 뒤 정제·대상 표를 계약대로 쓴다."""

from pathlib import Path

import pytest

from mlbproj import cli
from mlbproj import contracts as c
from mlbproj.interim import read_table
from mlbproj.validate import LeagueLimits, validate_seasons, write_report

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "raw"
RELAXED = LeagueLimits(
    pa_full=(0, 10**7),
    pa_short=(0, 10**7),
    outs_full=(0, 10**7),
    outs_short=(0, 10**7),
    pa_bf_identity=False,
)


def test_cli_build_writes_contract_tables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.json"
    write_report(validate_seasons(FIXTURE_RAW, c.SEASONS, limits=RELAXED), report)
    out = tmp_path / "interim"
    code = cli.main(
        ["build", "--raw-dir", str(FIXTURE_RAW), "--report", str(report), "--out-dir", str(out)]
    )
    assert code == 0
    for spec in (c.PLAYERS, c.BATTING_SEASONS, c.PITCHING_SEASONS, c.FIELDING_SEASONS, c.COHORTS):
        df = read_table(spec, c.interim_path(spec, out))
        assert len(df) > 0, spec.name
    cohorts = read_table(c.COHORTS, c.interim_path(c.COHORTS, out))
    assert sorted(cohorts["target_season"].unique().tolist()) == [2024, 2025, 2026, 2027]
    assert set(cohorts["role"]) == {"bat", "pit"}
    assert "cohorts:" in capsys.readouterr().out


def test_cli_build_is_byte_stable(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    write_report(validate_seasons(FIXTURE_RAW, c.SEASONS, limits=RELAXED), report)
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        cli.main(
            ["build", "--raw-dir", str(FIXTURE_RAW), "--report", str(report), "--out-dir", str(out)]
        )
    for spec in (c.PLAYERS, c.BATTING_SEASONS, c.PITCHING_SEASONS, c.FIELDING_SEASONS, c.COHORTS):
        assert c.interim_path(spec, first).read_bytes() == c.interim_path(spec, second).read_bytes()


def test_cli_build_refuses_without_passing_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "build",
            "--raw-dir",
            str(FIXTURE_RAW),
            "--report",
            str(tmp_path / "none.json"),
            "--out-dir",
            str(tmp_path / "o"),
        ]
    )
    assert code == 1
    assert "검증을 먼저 통과" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()
