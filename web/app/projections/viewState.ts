// 예측표 보기 상태 ↔ 주소 해시 (DESIGN §1-2) — 선수 상세에서 뒤로 가면 탭·검색·필터·정렬·쪽이 그대로 돌아온다.
// 해시는 기본값과 다른 것만 적는다: "#role=pit&q=ohtani&all=1&rp=1&career=1-2&page=3&sort=hr-desc" (rp=1: 투수 100이닝 필터 끔)
// 순수 함수만 둔다. 주소를 읽고 쓰는 일은 ProjectionsHome이 한다.
import type { CareerGroup, Role } from "@/src/data/types";
import {
  BAT_SORT_KEYS,
  DEFAULT_BAT_SORT,
  DEFAULT_PIT_SORT,
  PIT_SORT_KEYS,
  type BatSortKey,
  type CareerFilter,
  type PitSortKey,
  type SortEntry,
} from "./rows";

export interface ViewState {
  role: Role;
  query: string;
  /** "2026 출전 선수만" (DS3 — 켠 채 시작) */
  prevOnly: boolean;
  /** 투수 탭 "예상 100이닝 이상" (켠 채 시작, 2026-11-25 사용자 결정) */
  startersOnly: boolean;
  career: CareerFilter;
  /** 넓은 화면은 몇 쪽, 좁은 화면은 몇 묶음(50명씩)까지 보였는가 — 1부터 */
  page: number;
  batSort: SortEntry<BatSortKey>[];
  pitSort: SortEntry<PitSortKey>[];
}

/** 첫 화면 (DS2~DS4): 타자 탭, 2026 출전 선수만, 2027 OPS 높은 순, 1쪽 */
export const DEFAULT_VIEW: ViewState = {
  role: "bat",
  query: "",
  prevOnly: true,
  startersOnly: true,
  career: "all",
  page: 1,
  batSort: DEFAULT_BAT_SORT,
  pitSort: DEFAULT_PIT_SORT,
};

const CAREER_CODE: Record<CareerGroup, string> = { seasons_1_2: "1-2", seasons_3_5: "3-5" };
const VIEW_KEYS = ["role", "q", "all", "rp", "career", "page", "sort", "psort"] as const;

function encodeSort(sort: readonly SortEntry[]): string {
  const s = sort[0];
  return s === undefined ? "" : `${s.sortKey}-${s.direction === "ascending" ? "asc" : "desc"}`;
}

function decodeSort<K extends string>(raw: string | null, keys: readonly K[]): SortEntry<K>[] | null {
  const m = raw === null ? null : /^([A-Za-z]+)-(asc|desc)$/.exec(raw);
  if (m === null) return null;
  const key = keys.find((k) => k === m[1]);
  return key === undefined ? null : [{ sortKey: key, direction: m[2] === "asc" ? "ascending" : "descending" }];
}

/** 보기 상태 → 해시 본문("#" 없이). 모두 기본값이면 빈 문자열 */
export function encodeView(view: ViewState): string {
  const params = new URLSearchParams();
  if (view.role !== DEFAULT_VIEW.role) params.set("role", view.role);
  if (view.query !== "") params.set("q", view.query);
  if (!view.prevOnly) params.set("all", "1");
  if (!view.startersOnly) params.set("rp", "1");
  if (view.career !== "all") params.set("career", CAREER_CODE[view.career]);
  if (view.page > 1) params.set("page", String(view.page));
  const bat = encodeSort(view.batSort);
  const pit = encodeSort(view.pitSort);
  if (bat !== encodeSort(DEFAULT_BAT_SORT)) params.set("sort", bat);
  if (pit !== encodeSort(DEFAULT_PIT_SORT)) params.set("psort", pit);
  return params.toString();
}

/** 해시가 보기 상태를 담았는가 — "#projections" 같은 앵커는 아니다 */
export function isViewHash(hash: string): boolean {
  const body = hash.replace(/^#/, "");
  if (!body.includes("=")) return false;
  const params = new URLSearchParams(body);
  return VIEW_KEYS.some((k) => params.has(k));
}

/** 해시 → 보기 상태. 보기 상태가 아니면 null, 틀린 값은 그 칸만 기본값으로 */
export function decodeView(hash: string): ViewState | null {
  if (!isViewHash(hash)) return null;
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const career = (Object.keys(CAREER_CODE) as CareerGroup[]).find((g) => CAREER_CODE[g] === params.get("career"));
  const pageRaw = params.get("page");
  const page = pageRaw !== null && /^\d{1,4}$/.test(pageRaw) ? Number(pageRaw) : 1;
  return {
    role: params.get("role") === "pit" ? "pit" : "bat",
    query: params.get("q") ?? "",
    prevOnly: params.get("all") !== "1",
    startersOnly: params.get("rp") !== "1",
    career: career ?? "all",
    page: page >= 1 ? page : 1,
    batSort: decodeSort(params.get("sort"), BAT_SORT_KEYS) ?? DEFAULT_BAT_SORT,
    pitSort: decodeSort(params.get("psort"), PIT_SORT_KEYS) ?? DEFAULT_PIT_SORT,
  };
}
