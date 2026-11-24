"""정제 표 CSV 입출력 (SPEC N1, PLAN §2-1 `data/interim`).

표는 계약(`contracts.TableSpec`)을 지킨 채로만 쓰고 읽는다. 쓸 때는 키 순서로 정렬해
같은 표면 언제나 같은 바이트가 나오게 한다. 실수는 왕복해도 값이 바뀌지 않게 쓴다.
"""

import os
from pathlib import Path
from typing import Any

import pandas as pd

from mlbproj.contracts import Column, TableSpec, check_frame


def _dtype(col: Column) -> str:
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    if col.kind == "bool":
        return "boolean" if col.nullable else "bool"
    return "str"


_BOOL_TEXT = {True: True, False: False, "True": True, "False": False}


def _read_bool(series: pd.Series, col: Column, path: Path) -> pd.Series:
    """참·거짓 열을 읽는다. 빈 칸을 참으로 바꾸는 `astype(bool)`을 쓰지 않는다."""
    present = series.notna()
    unknown = sorted({str(v) for v in series[present] if v not in _BOOL_TEXT})
    if unknown:
        raise ValueError(f"{path}: {col.name} 열에 참·거짓이 아닌 값 {unknown}")
    if not col.nullable and not bool(present.all()):
        raise ValueError(f"{path}: {col.name} 열에 빈 칸 (결측 허용 안 됨)")
    values = [_BOOL_TEXT[v] if pd.notna(v) else pd.NA for v in series]
    dtype: Any = _dtype(col)
    return pd.Series(values, index=series.index, dtype=dtype, name=series.name)


def write_table(df: pd.DataFrame, spec: TableSpec, path: Path) -> None:
    """계약을 확인한 뒤 키 순서로 정렬해 CSV로 쓴다.

    Args:
        df: 쓸 표.
        spec: 계약.
        path: CSV 경로. 상위 디렉토리는 만든다.

    Raises:
        ValueError: 표가 계약을 어길 때 (파일을 쓰지 않는다).
    """
    problems = check_frame(df, spec)
    if problems:
        raise ValueError("; ".join(problems))
    out = df.sort_values(list(spec.keys), kind="mergesort").reset_index(drop=True)
    text = out.to_csv(index=False, lineterminator="\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_table(spec: TableSpec, path: Path) -> pd.DataFrame:
    """`write_table`이 쓴 CSV를 계약의 형식으로 읽는다.

    Args:
        spec: 계약.
        path: CSV 경로.

    Returns:
        계약을 지키는 표.

    Raises:
        ValueError: 머리줄이 계약과 다르거나, 읽은 표가 계약을 어길 때.
    """
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if header != spec.column_names:
        raise ValueError(f"{path}: 열이 계약({spec.name})과 다르다 — {header}")
    df = pd.read_csv(
        path, dtype={col.name: _dtype(col) for col in spec.columns if col.kind != "bool"}
    )
    for col in spec.columns:
        if col.kind == "bool":
            df[col.name] = _read_bool(df[col.name], col, path)
    problems = check_frame(df, spec)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return df
