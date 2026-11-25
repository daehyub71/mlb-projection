// WAR 순위 행 모델 (F21, SA-6 · SA-9) — war.json 행을 표·카드가 쓰는 평평한 행으로 바꾸고, 정렬하고, 거른다.
// 순수 함수만 둔다. 숫자 표기는 여기서 하지 않는다(화면에서 src/format.ts로).
import type { WarFile, WarRoles } from "@/src/data/types";
import { fmtCount } from "@/src/format";
import { normalizeName } from "../rows";
import type { WarSort } from "../viewState";

export interface WarView extends Record<string, unknown> {
  id: number;
  name: string;
  /** 검색용 이름 — 소문자, 악센트 제거 (예측표 검색과 같은 규칙) */
  searchName: string;
  position: string | null;
  team: string | null;
  active: boolean;
  roles: WarRoles;
  pa: number | null;
  ip: number | null;
  war: number;
  warBat: number | null;
  warPit: number | null;
  warFull: number;
  prevWar: number | null;
  /** 정렬 뒤 전체 순위(1부터) — 검색으로 걸러도 전체 순위를 그대로 둔다 */
  rank: number;
}

export function toWarViews(file: WarFile): WarView[] {
  return file.rows.map((row) => ({
    id: row.player.player_id,
    name: row.player.name,
    searchName: normalizeName(row.player.name),
    position: row.player.primary_position,
    team: row.player.team_name,
    active: row.player.active,
    roles: row.roles,
    pa: row.pa,
    ip: row.ip,
    war: row.war,
    warBat: row.war_bat,
    warPit: row.war_pit,
    warFull: row.war_full,
    prevWar: row.prev_war,
    rank: 0,
  }));
}

/** 높은 순, 같으면 이름순 — 순위 번호를 새로 매긴다 */
export function rankWar(views: readonly WarView[], sort: WarSort): WarView[] {
  const value = (v: WarView) => (sort === "full" ? v.warFull : v.war);
  return [...views]
    .sort((a, b) => value(b) - value(a) || a.name.localeCompare(b.name, "en") || a.id - b.id)
    .map((v, i) => ({ ...v, rank: i + 1 }));
}

/** 이름 검색 — 대소문자·악센트를 무시한다 */
export function filterWar(views: readonly WarView[], query: string): WarView[] {
  const term = normalizeName(query);
  return term === "" ? [...views] : views.filter((v) => v.searchName.includes(term));
}

/** 예측 출전 "594타석", "132이닝", 투타 "500타석 · 77이닝" */
export function playingTime(v: Pick<WarView, "pa" | "ip">): string {
  const parts = [v.pa === null ? null : `${fmtCount(v.pa)}타석`, v.ip === null ? null : `${fmtCount(v.ip)}이닝`].filter(
    (p): p is string => p !== null,
  );
  return parts.join(" · ");
}

/** "CF · Chicago Cubs" — 빈 값은 빼고, 다 비면 "" (화면에서 —로) */
export function warPlayerMeta(v: Pick<WarView, "position" | "team">): string {
  return [v.position, v.team].filter((p): p is string => p !== null && p !== "").join(" · ");
}
