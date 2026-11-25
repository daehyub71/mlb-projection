// WAR 화면 공통 계산 — 순수 함수만 (메인 세션만 고친다)
import type { CareerGroup, Role, WarEvalRow, WarFile, WarMetric, WarRow } from "./data/types";

/**
 * 그룹 평균 오차(승, D39): evaluations에서 (role, metric, group) 행의 mae를 목표 시즌 전체에 걸쳐 단순 평균.
 * 기존 지표의 group_error와 같은 규칙(mlbproj/export.py). 값이 없으면 null.
 */
export function warGroupMae(
  evaluations: readonly WarEvalRow[],
  role: Role,
  group: CareerGroup,
  metric: WarMetric = "war",
): number | null {
  const values = evaluations
    .filter((row) => row.role === role && row.metric === metric && row.group === group && row.mae !== null)
    .map((row) => row.mae as number);
  return values.length === 0 ? null : values.reduce((sum, v) => sum + v, 0) / values.length;
}

/** 선수 ID → war.json 행. 2026에 뛰지 않은 선수는 없다(D42) → null */
export function findWarRow(war: WarFile, playerId: number): WarRow | null {
  return war.rows.find((row) => row.player.player_id === playerId) ?? null;
}
