"""명령줄 진입점 (PLAN §2-4). 명령은 마일스톤마다 붙인다."""

import argparse

from mlbproj import __version__


def build_parser() -> argparse.ArgumentParser:
    """명령줄 파서를 만든다.

    Returns:
        하위 명령을 담은 파서.
    """
    parser = argparse.ArgumentParser(
        prog="mlbproj",
        description="재미로 보는 MLB 선수 성적 예측 — 수집 → 검증 → 정제 → 예측 → 내보내기",
    )
    parser.add_argument("--version", action="version", version=f"mlbproj {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: list[str] | None = None) -> int:
    """명령을 실행한다.

    Args:
        argv: 인자 목록. None이면 sys.argv를 쓴다.

    Returns:
        종료 코드.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0
