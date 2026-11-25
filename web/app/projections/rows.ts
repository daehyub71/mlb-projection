// 예측표 행 모델 — JSON 행(BatRow·PitRow)을 표·카드가 쓰는 평평한 행으로 바꾸고, 거르고, 정렬한다 (F13, DS2~DS4)
// 순수 함수만 둔다. 숫자 표기는 여기서 하지 않는다(화면에서 src/format.ts로).
import type { BatTable, CareerGroup, PitTable } from "@/src/data/types";
import { careerGroup } from "@/src/site";

export type CareerFilter = "all" | CareerGroup;
export type SortDirection = "ascending" | "descending";
export interface SortEntry<K extends string = string> {
  sortKey: K;
  direction: SortDirection;
}

/** 두 역할이 같이 쓰는 칸 */
interface BaseView extends Record<string, unknown> {
  id: number;
  name: string;
  /** 검색용 이름 — 소문자, 악센트 제거 */
  searchName: string;
  position: string | null;
  team: string | null;
  age: number | null;
  active: boolean;
  lastSeason: number;
  nSeasons: number;
  career: CareerGroup;
  /** 직전 시즌(2026)에 뛰었는가 — "2026 출전 선수만" 필터 (DS3) */
  playedPrev: boolean;
}

export interface BatView extends BaseView {
  samplePa: number;
  ops: number;
  opsBand: number | null;
  /** 2027 OPS − 리그 평균. 리그 평균이 없으면 null */
  opsDiff: number | null;
  pa: number;
  hr: number;
  prevOps: number | null;
}

export interface PitView extends BaseView {
  sampleIp: number;
  era: number;
  eraBand: number | null;
  kbb: number;
  kbbBand: number | null;
  ip: number;
  prevEra: number | null;
}

/** 정렬할 수 있는 열 — 표 열 key와 같다 (주소 해시를 읽을 때 이 목록으로 검사한다) */
export const BAT_SORT_KEYS = ["name", "age", "samplePa", "ops", "pa", "hr", "prevOps"] as const;
export const PIT_SORT_KEYS = ["name", "age", "sampleIp", "era", "kbb", "ip", "prevEra"] as const;
export type BatSortKey = (typeof BAT_SORT_KEYS)[number];
export type PitSortKey = (typeof PIT_SORT_KEYS)[number];

/** 첫 화면 정렬 — 타자 2027 OPS 높은 순(DS2), 투수 2027 ERA 낮은 순 */
export const DEFAULT_BAT_SORT: SortEntry<BatSortKey>[] = [{ sortKey: "ops", direction: "descending" }];
export const DEFAULT_PIT_SORT: SortEntry<PitSortKey>[] = [{ sortKey: "era", direction: "ascending" }];

/** 새 열을 처음 누를 때의 방향 — 좋은 쪽이 위로 (ERA·나이·이름은 낮은 순, 나머지는 높은 순) */
const NATURAL_DIRECTION: Record<BatSortKey | PitSortKey, SortDirection> = {
  name: "ascending",
  age: "ascending",
  samplePa: "descending",
  sampleIp: "descending",
  ops: "descending",
  pa: "descending",
  hr: "descending",
  prevOps: "descending",
  era: "ascending",
  kbb: "descending",
  ip: "descending",
  prevEra: "ascending",
};

/** 검색 비교용 — 대소문자·악센트를 무시한다 ("Acuña" ↔ "acuna") */
export function normalizeName(text: string): string {
  return text.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().trim();
}

function finite(v: number | null | undefined): number | null {
  return v == null || !Number.isFinite(v) ? null : v;
}

function baseView(
  player: BatTable["rows"][number]["player"],
  nSeasons: number,
  playedPrev: boolean,
): BaseView {
  return {
    id: player.player_id,
    name: player.name,
    searchName: normalizeName(player.name),
    position: player.primary_position,
    team: player.team_name,
    age: finite(player.age),
    active: player.active,
    lastSeason: player.last_mlb_season,
    nSeasons,
    career: careerGroup(nSeasons),
    playedPrev,
  };
}

export function toBatViews(table: BatTable): BatView[] {
  const league = finite(table.league_baseline.ops);
  return table.rows.map((row) => {
    const base = baseView(row.player, row.window.n_seasons, row.prev !== null);
    const ops = row.projection.ops;
    return {
      ...base,
      samplePa: row.sample_pa,
      ops,
      opsBand: finite(table.bands[base.career]?.ops),
      opsDiff: league === null ? null : ops - league,
      pa: row.projection.pa,
      hr: row.projection.hr,
      prevOps: finite(row.prev?.ops),
    };
  });
}

export function toPitViews(table: PitTable): PitView[] {
  return table.rows.map((row) => {
    const base = baseView(row.player, row.window.n_seasons, row.prev !== null);
    const band = table.bands[base.career];
    return {
      ...base,
      sampleIp: row.sample_ip,
      era: row.projection.era,
      eraBand: finite(band?.era),
      kbb: row.projection.k_bb_pct,
      kbbBand: finite(band?.k_bb_pct),
      ip: row.projection.ip,
      prevEra: finite(row.prev?.era),
    };
  });
}

/** 투수 탭 기본 필터 '예상 100이닝 이상' — 불펜 소이닝 투수가 ERA 순위 위를 차지하지 않게 (2026-11-25 사용자 결정) */
export const PIT_MIN_IP = 100;

/** 예상 이닝이 PIT_MIN_IP 이상인 투수만 (on이 false면 그대로) */
export function filterStarters<T extends { ip: number }>(rows: readonly T[], on: boolean): T[] {
  return on ? rows.filter((r) => r.ip >= PIT_MIN_IP) : [...rows];
}

export interface RowFilter {
  query: string;
  prevOnly: boolean;
  career: CareerFilter;
}

/** 검색(부분 일치) · 2026 출전 · 사용 시즌 그룹을 모두 만족하는 행 (AND) */
export function filterRows<T extends BaseView>(rows: readonly T[], filter: RowFilter): T[] {
  const term = normalizeName(filter.query);
  return rows.filter(
    (r) =>
      (!filter.prevOnly || r.playedPrev) &&
      (filter.career === "all" || r.career === filter.career) &&
      (term === "" || r.searchName.includes(term)),
  );
}

/** 정렬 — 빈 값(null)은 방향과 상관없이 맨 뒤, 같은 값은 이름순 */
export function sortRows<T extends BaseView>(rows: readonly T[], sort: readonly SortEntry[]): T[] {
  const entry = sort[0];
  const byName = (a: T, b: T) => a.name.localeCompare(b.name, "en") || a.id - b.id;
  if (entry === undefined) return [...rows].sort(byName);
  const sign = entry.direction === "ascending" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const va = a[entry.sortKey];
    const vb = b[entry.sortKey];
    const aMissing = va == null || (typeof va === "number" && !Number.isFinite(va));
    const bMissing = vb == null || (typeof vb === "number" && !Number.isFinite(vb));
    if (aMissing || bMissing) return aMissing === bMissing ? byName(a, b) : aMissing ? 1 : -1;
    const cmp =
      typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb), "en");
    return cmp === 0 ? byName(a, b) : sign * cmp;
  });
}

/** 머리를 누른 결과를 받아, 새 열이면 그 열의 자연스러운 방향으로 시작한다 */
export function nextSort<K extends BatSortKey | PitSortKey>(
  current: readonly SortEntry<K>[],
  requested: readonly SortEntry<K>[],
): SortEntry<K>[] {
  const next = requested[0];
  if (next === undefined) return [...current];
  if (current[0]?.sortKey === next.sortKey) return [next];
  return [{ sortKey: next.sortKey, direction: NATURAL_DIRECTION[next.sortKey] }];
}

/** 쪽 수 (DS4) — 행이 없어도 1쪽 */
export function pageCount(total: number, pageSize: number): number {
  return Math.max(1, Math.ceil(total / pageSize));
}
