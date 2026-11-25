// WAR 순위 (F21, DESIGN SA-6 데스크톱 · SA-9 모바일, SPEC D33~D35 · D39 · D41 · D42) — M9 레인 A Red 테스트
// 예측표의 세 번째 탭 「WAR 순위」. 데이터는 가짜 fetch로 준다(네트워크 없음). 숫자 기대값은 모두 src/format.ts 함수로 만든다.
// 선수 이름은 모두 지어낸 것이다 — 실존 선수 이름을 쓰지 않는다. war.json 모양은 합성 샘플(tests/fixtures/synthetic/web/war.json)에서 가져온다.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import HomePage from "@/app/page";
import { Providers } from "@/app/providers";
import { decodeView, encodeView, isViewHash } from "@/app/projections/viewState";
import { DATA_BASE, parsePlayerId } from "@/src/data/load";
import type { BatTable, CareerGroup, WarEvalRow, WarFile, WarRoles, WarRow, WarSlot } from "@/src/data/types";
import { DASH, fmtCount, fmtWar, fmtWarSplit, warRoleLabel } from "@/src/format";
import { initials, playerPhotoUrl } from "@/src/photo";
import { BAND_NOTE, PAGE_SIZE, PLAYING_TIME_NOTE, playerHref } from "@/src/site";
import { warGroupMae } from "@/src/war";
import { FULL_TIME_ADJUSTED, WAR_ABOUT, WAR_CONSERVATIVE, WAR_PREV_ONLY, fullTimeUnits } from "@/src/warText";

// ---------------------------------------------------------------------------
// 픽스처 — 합성 샘플의 meta·full_time을 바탕으로 손으로 만든 WAR 표
// ---------------------------------------------------------------------------

const ROOT = join(import.meta.dirname, "..", "..");
const sample = <T,>(name: string): T =>
  JSON.parse(readFileSync(join(ROOT, "tests/fixtures/synthetic/web", name), "utf-8")) as T;
const sampleBat = sample<BatTable>("batters.json");
const sampleWar = sample<WarFile>("war.json");

/** 첫 화면(타자 탭) 표 — 둘째 선수(401)는 예측표에는 있지만 2026에 뛰지 않아 WAR 순위에 없다 (D42) */
const BAT: BatTable = {
  ...sampleBat,
  rows: [
    sampleBat.rows[0]!,
    {
      ...sampleBat.rows[0]!,
      player: { ...sampleBat.rows[0]!.player, player_id: 401, name: "Bench Only", last_mlb_season: 2025 },
      prev: null,
    },
  ],
};

interface WarSpec {
  id: number;
  name: string;
  pos?: string | null;
  team?: string | null;
  active?: boolean;
  roles: WarRoles;
  group?: CareerGroup;
  slot?: WarSlot;
  pa?: number;
  ip?: number;
  bat?: number;
  pit?: number;
  fullBat?: number;
  fullPit?: number;
  prev: number | null;
}

/** JSON은 소수 2자리 */
const r2 = (v: number): number => Math.round(v * 100) / 100;

function warRow(s: WarSpec): WarRow {
  const bat = s.bat ?? null;
  const pit = s.pit ?? null;
  const fullBat = s.fullBat ?? null;
  const fullPit = s.fullPit ?? null;
  return {
    player: {
      player_id: s.id,
      name: s.name,
      primary_position: s.pos === undefined ? (s.roles === "pit" ? "P" : "1B") : s.pos,
      age: 28,
      team_name: s.team === undefined ? "Sample Club" : s.team,
      active: s.active ?? true,
      last_mlb_season: 2026,
    },
    roles: s.roles,
    played_prev: true,
    group: s.group ?? "seasons_3_5",
    slot: s.roles === "bat" ? null : (s.slot ?? "sp"),
    pa: s.pa ?? null,
    ip: s.ip ?? null,
    war_bat: bat,
    war_pit: pit,
    war: r2((bat ?? 0) + (pit ?? 0)),
    war_full_bat: fullBat,
    war_full_pit: fullPit,
    war_full: r2((fullBat ?? 0) + (fullPit ?? 0)),
    prev_war: s.prev,
    prev_bat_runs: null,
    history: [
      {
        season: 2026,
        war_bat: s.roles === "pit" ? null : 1,
        war_pit: s.roles === "bat" ? null : 1,
        war_pit_ra9: s.roles === "bat" ? null : 1,
      },
    ],
  };
}

function ev(target: number, role: "bat" | "pit", metric: "war" | "war_full", group: WarEvalRow["group"], mae: number): WarEvalRow {
  return { target_season: target, role, metric, group, n: 100, mae, baseline_mae_prev: mae + 0.1, baseline_mae_league: mae + 0.05 };
}

/**
 * 과거 검증 행 — 문장에 쓰는 칸(3~5시즌 · 예측 WAR)은 시즌마다 다르게 둬서 화면이 평균(warGroupMae)을 쓰는지 본다.
 * 타자 3~5시즌: .74 · .50 → 평균 .62 ("0.6", 한 시즌만 쓰면 0.7·0.5), 투수 3~5시즌: .47 · .33 → .40 ("0.4", 한 시즌만이면 0.5·0.3).
 * 나머지는 문장에 쓰지 않는 그룹·지표 — 섞이면 숫자가 달라진다.
 */
const EVALS: WarEvalRow[] = [
  ev(2024, "bat", "war", "seasons_3_5", 0.74),
  ev(2025, "bat", "war", "seasons_3_5", 0.5),
  ev(2024, "pit", "war", "seasons_3_5", 0.47),
  ev(2025, "pit", "war", "seasons_3_5", 0.33),
  ev(2024, "bat", "war", "all", 0.9),
  ev(2024, "bat", "war", "seasons_1_2", 0.2),
  ev(2024, "bat", "war_full", "seasons_3_5", 1.4),
  ev(2024, "pit", "war", "all", 0.8),
];

function warFile(rows: WarRow[], evaluations: WarEvalRow[] = EVALS): WarFile {
  return { ...sampleWar, rows, evaluations, final_test: null };
}

// JSON 순서는 일부러 섞었다 — 화면이 직접 정렬해야 한다.
// 예측 WAR 동점: 306 · 307 (4.3) → 이름순 "Tie Breaker A"(307)가 먼저. 풀타임 환산 동점: 301 · 303 (6.63) → "Alpha Center"(301)가 먼저.
const WAR_SPECS: WarSpec[] = [
  { id: 308, name: "Negative Value", pos: null, team: null, roles: "bat", pa: 180.4, bat: -0.35, fullBat: -0.42, prev: -0.3 },
  { id: 304, name: "Ace Starter", roles: "pit", slot: "sp", ip: 131.9, pit: 4.26, fullPit: 5.81, prev: 6.65 },
  {
    id: 302,
    name: "Two Way Star",
    pos: "TWP",
    team: "Sample Coast",
    roles: "both",
    slot: "sp",
    pa: 499.8,
    ip: 76.7,
    bat: 3.9,
    pit: 2.28,
    fullBat: 4.68,
    fullPit: 5.35,
    prev: 6.63,
  },
  { id: 306, name: "Tie Breaker B", pos: "2B", roles: "bat", pa: 560.2, bat: 4.3, fullBat: 4.9, prev: 3.8 },
  { id: 301, name: "Alpha Center", pos: "CF", team: "Sample Lakes", roles: "bat", pa: 593.5, bat: 6.56, fullBat: 6.63, prev: 10.62 },
  { id: 305, name: "Bullpen Arm", roles: "pit", slot: "rp", group: "seasons_1_2", ip: 60.3, pit: 0.9, fullPit: 1.2, prev: null },
  { id: 309, name: "José Ejemplo", pos: "LF", roles: "bat", pa: 540, bat: 3.1, fullBat: 3.5, prev: 2.9 },
  { id: 303, name: "Short Season Slugger", pos: "RF", roles: "bat", pa: 252.4, bat: 2.77, fullBat: 6.63, prev: 4.85 },
  { id: 307, name: "Tie Breaker A", pos: "SS", roles: "bat", pa: 548.9, bat: 4.3, fullBat: 4.6, prev: 5.12 },
];
const WAR = warFile(WAR_SPECS.map(warRow));
const warOf = (id: number): WarRow => WAR.rows.find((r) => r.player.player_id === id)!;

/** 예측 WAR 높은 순(기본) — 같으면 이름순 */
const WAR_ORDER = [301, 302, 307, 306, 304, 309, 303, 305, 308];
/** 풀타임 환산 WAR 높은 순 — 같으면 이름순 */
const FULL_ORDER = [302, 301, 303, 304, 306, 307, 309, 305, 308];

/** 한 쪽(50행)을 넘는 큰 WAR 표 — 셋째마다 투수, 예측 WAR은 번호 순, 풀타임 환산은 다른 순서(37을 곱한 나머지) */
function bigWar(count: number): WarFile {
  return warFile(
    Array.from({ length: count }, (_, i) => {
      const isPit = i % 3 === 2;
      const war = r2(8 - i * 0.01);
      const full = r2(((i * 37) % count) * 0.01);
      return warRow({
        id: 5001 + i,
        name: `War Player ${String(i + 1).padStart(4, "0")}`,
        roles: isPit ? "pit" : "bat",
        pa: isPit ? undefined : 300 + (i % 200),
        ip: isPit ? 50 + (i % 120) : undefined,
        bat: isPit ? undefined : war,
        pit: isPit ? war : undefined,
        fullBat: isPit ? undefined : full,
        fullPit: isPit ? full : undefined,
        prev: 1,
      });
    }),
  );
}

/** 정렬 규칙대로 기대 순서를 만든다 — 높은 순, 같으면 이름순 */
function orderBy(file: WarFile, key: "war" | "war_full"): number[] {
  return [...file.rows]
    .sort((a, b) => b[key] - a[key] || a.player.name.localeCompare(b.player.name, "en"))
    .map((r) => r.player.player_id);
}

/** 예측 출전 표기 — 타석은 fmtCount+"타석", 이닝은 fmtCount+"이닝", 투타는 "500타석 · 77이닝" */
function playingTimePattern(row: WarRow): RegExp {
  const parts = [
    row.pa === null ? null : `${fmtCount(row.pa)}\\s*타석`,
    row.ip === null ? null : `${fmtCount(row.ip)}\\s*이닝`,
  ].filter((p): p is string => p !== null);
  return new RegExp(parts.join("\\s*·\\s*"));
}

// ---------------------------------------------------------------------------
// 가짜 fetch · 화면 너비
// ---------------------------------------------------------------------------

const BAT_URL = `${DATA_BASE}/batters.json`;
const WAR_URL = `${DATA_BASE}/war.json`;

type Served = unknown | (() => Promise<Response>);

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

/** 경로 → 응답. 함수면 그 함수가 응답을 만든다. 모르는 경로는 404 */
function stubData(files: Record<string, Served>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
    const raw = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const path = raw.replace(/^https?:\/\/[^/]+/, "");
    const body = files[path];
    if (body === undefined) return new Response("not found", { status: 404 });
    if (typeof body === "function") return (body as () => Promise<Response>)();
    return jsonResponse(body);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function callsTo(fetchMock: ReturnType<typeof stubData>, url: string): number {
  return fetchMock.mock.calls.filter(([input]) => String(input instanceof Request ? input.url : input).endsWith(url))
    .length;
}

/** matchMedia를 화면 너비 기준으로 흉내 낸다 — min/max-width(px·em·rem)와 범위 문법 */
function mediaMatches(query: string, width: number): boolean {
  return query.split(",").some((part) => {
    const p = part.trim().toLowerCase();
    if (p.startsWith("not ")) return !mediaMatches(p.slice(4), width);
    return p
      .replace(/^only\s+/, "")
      .split(/\s+and\s+/)
      .every((cond) => {
        const c = cond.trim();
        if (c === "all" || c === "screen") return true;
        const toPx = (v: string, unit: string) => Number(v) * (unit === "px" ? 1 : 16);
        let m = /^\(\s*(min|max)-width\s*:\s*([\d.]+)(px|em|rem)\s*\)$/.exec(c);
        if (m) return m[1] === "min" ? width >= toPx(m[2]!, m[3]!) : width <= toPx(m[2]!, m[3]!);
        m = /^\(\s*width\s*(>=|<=|>|<)\s*([\d.]+)(px|em|rem)\s*\)$/.exec(c);
        if (m) {
          const px = toPx(m[2]!, m[3]!);
          return { ">=": width >= px, "<=": width <= px, ">": width > px, "<": width < px }[m[1] as ">=" | "<=" | ">" | "<"];
        }
        return false; // pointer·hover·prefers-* 등은 맞지 않는 것으로
      });
  });
}

function setViewport(width: number) {
  vi.stubGlobal("innerWidth", width);
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: mediaMatches(query, width),
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  }));
}

beforeEach(() => {
  setViewport(1280);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  // 예측표는 보기 상태를 주소 해시에 적는다 — 다음 테스트로 새지 않게 주소를 되돌린다
  window.history.replaceState(null, "", "/");
});

// ---------------------------------------------------------------------------
// 화면 읽기 도우미 (projections.test.tsx와 같은 방식)
// ---------------------------------------------------------------------------

const norm = (s: string | null | undefined): string => (s ?? "").replace(/\s+/g, " ").trim();

function renderHome() {
  return render(
    <Providers>
      <HomePage />
    </Providers>,
  );
}

/** 처음 들어오는 주소 — 선수 상세에서 뒤로 가기로 돌아온 경우를 흉내 낸다 */
const arriveAt = (url: string) => window.history.replaceState(null, "", url);

const findTable = () => screen.findByRole("table", {}, { timeout: 3000 });
const getTable = () => screen.getByRole("table");

// next/link는 테스트에서 끝의 /를 뗀다(/player?id=1) — 빌드는 trailingSlash라 /player/?id=1. 둘 다 같은 주소로 본다
const canonHref = (href: string | null | undefined): string => (href ?? "").replace(/\/(?=[?#]|$)/, "");

function playerLink(el: Element): Element | null {
  return el.querySelector('a[href*="/player"][href*="id="]');
}

function expectHref(el: Element, href: string) {
  expect(canonHref(el.getAttribute("href"))).toBe(canonHref(href));
}

function rowIdOf(row: Element): number {
  const href = playerLink(row)!.getAttribute("href")!;
  return parsePlayerId(href.slice(href.indexOf("?"))) ?? -1;
}

/** 머리 행을 뺀, 선수 링크가 있는 표의 행 */
function dataRows(table: HTMLElement): HTMLElement[] {
  return within(table)
    .getAllByRole("row")
    .filter((r) => playerLink(r) !== null);
}

const rowIds = (table: HTMLElement): number[] => dataRows(table).map(rowIdOf);

function rowOf(table: HTMLElement, id: number): HTMLElement {
  const row = dataRows(table).find((r) => rowIdOf(r) === id);
  expect(row, `선수 ${id}의 행`).toBeDefined();
  return row!;
}

function headers(table: HTMLElement): HTMLElement[] {
  return within(table).getAllByRole("columnheader");
}

function headerOf(table: HTMLElement, re: RegExp): HTMLElement {
  const h = headers(table).find((el) => re.test(norm(el.textContent)));
  expect(h, `열 머리 ${re}`).toBeDefined();
  return h!;
}

/** 열 머리 순서로 그 행의 칸을 찾는다 */
function cellOf(table: HTMLElement, row: HTMLElement, re: RegExp): HTMLElement {
  const idx = headers(table).indexOf(headerOf(table, re));
  const cell = row.children[idx];
  expect(cell, `${re} 칸`).toBeDefined();
  return cell as HTMLElement;
}

const cellText = (table: HTMLElement, id: number, re: RegExp): string => norm(cellOf(table, rowOf(table, id), re).textContent);

/** 열 머리 (SA-6) */
const COL = {
  rank: /^(#|순위)/,
  player: /^선수/,
  role: /^역할/,
  war: /^2027 예측 WAR\*/,
  full: /^풀타임 환산/,
  prev: /^2026 WAR/,
  pt: /^예측 출전\*/,
} as const;

/** 예측표 탭 — Astryx TabList(버튼+aria-current)·tab·radio 무엇이든 */
function roleTab(label: "타자" | "투수" | "WAR 순위"): HTMLElement {
  const name = new RegExp(`^${label}`);
  const found = [
    ...screen.queryAllByRole("tab", { name }),
    ...screen.queryAllByRole("radio", { name }),
    ...screen.queryAllByRole("button", { name }),
    ...screen.queryAllByRole("menuitemradio", { name }),
  ];
  expect(found.length, `예측표 탭 "${label}"`).toBeGreaterThan(0);
  return found[0]!;
}

function isSelected(el: HTMLElement): boolean {
  const current = el.getAttribute("aria-current");
  return (
    el.getAttribute("aria-selected") === "true" ||
    el.getAttribute("aria-checked") === "true" ||
    el.getAttribute("aria-pressed") === "true" ||
    (current !== null && current !== "false") ||
    (el instanceof HTMLInputElement && el.checked)
  );
}

/** 정렬 세그먼트 (SA-6 「예측 WAR」/「풀타임 환산 WAR」, SA-9 「예측 WAR」/「풀타임 환산」) */
const SEG_WAR = /^예측 WAR/;
const SEG_FULL = /^풀타임 환산/;

function sortSegment(name: RegExp): HTMLElement {
  const found = screen.queryAllByRole("radio", { name });
  expect(found.length, `정렬 세그먼트 ${name}`).toBeGreaterThan(0);
  return found[0]!;
}

function searchBox(): HTMLInputElement {
  const name = /선수 이름 검색/;
  const found = [...screen.queryAllByRole("searchbox", { name }), ...screen.queryAllByRole("textbox", { name })];
  expect(found.length, "선수 이름 검색 입력").toBeGreaterThan(0);
  return found[0] as HTMLInputElement;
}

const search = (text: string) => fireEvent.change(searchBox(), { target: { value: text } });

function pageNav(): HTMLElement {
  const navs = screen.queryAllByRole("navigation", { name: /페이지|pagination/i });
  expect(navs.length, "페이지 이동").toBeGreaterThan(0);
  return navs[0]!;
}

function goToPage(n: number) {
  const nav = pageNav();
  const re = new RegExp(`(^|\\D)${n}(\\D|$)`);
  const target = [...within(nav).queryAllByRole("button"), ...within(nav).queryAllByRole("link")].find((el) =>
    re.test(el.getAttribute("aria-label") ?? norm(el.textContent)),
  );
  expect(target, `${n}쪽 단추`).toBeDefined();
  fireEvent.click(target!);
}

/** 사진 대신 이니셜이 보이는지 — 글자로 그렸거나, Astryx Avatar처럼 CSS content로 그렸거나 */
function showsInitials(cell: HTMLElement, text: string): boolean {
  if (within(cell).queryAllByText(text).length > 0) return true;
  return Array.from(cell.querySelectorAll("[style]")).some((el) => (el.getAttribute("style") ?? "").includes(`"${text}"`));
}

function mlbPhotos(root: ParentNode): HTMLImageElement[] {
  return Array.from(root.querySelectorAll("img")).filter((img) =>
    (img.getAttribute("src") ?? "").startsWith("https://img.mlbstatic.com/"),
  );
}

/** 그 글을 담은 가장 안쪽 요소들 (글이 여러 요소로 나뉘어도 된다) */
function blocksWith(text: string): HTMLElement[] {
  return screen.queryAllByText((_, el) => {
    if (el === null || !norm(el.textContent).includes(text)) return false;
    return Array.from(el.children).every((c) => !norm(c.textContent).includes(text));
  });
}

const follows = (a: Node, b: Node): boolean => (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0;

/** 오류 안내가 보이는가 — Astryx 공용 알림 영역(role=alert)은 늘 있으므로 role만으로는 세지 않고 글자를 본다 */
function warErrorShown(): boolean {
  return blocksWith("WAR 순위를 불러오지 못했습니다").length > 0;
}

/** WAR 탭을 누르고 순위표가 기대 순서로 뜰 때까지 기다린다 */
async function openWarTab(expected: number[] = WAR_ORDER): Promise<HTMLElement> {
  fireEvent.click(roleTab("WAR 순위"));
  await waitFor(() => expect(rowIds(getTable())).toEqual(expected), { timeout: 3000 });
  return getTable();
}

/** 좁은 화면의 선수 카드 목록 — 선수 링크가 가장 많은 목록 */
function cardList(): HTMLElement {
  const lists = screen
    .queryAllByRole("list")
    .map((l) => ({ l, n: within(l).queryAllByRole("listitem").filter((li) => playerLink(li) !== null).length }))
    .filter((x) => x.n > 0)
    .sort((a, b) => b.n - a.n);
  expect(lists.length, "선수 카드 목록 (role=list)").toBeGreaterThan(0);
  return lists[0]!.l;
}

function cards(): HTMLElement[] {
  return within(cardList())
    .getAllByRole("listitem")
    .filter((li) => playerLink(li) !== null);
}

async function cardIds(): Promise<number[]> {
  return waitFor(() => cards().map(rowIdOf), { timeout: 3000 });
}

/** 데스크톱 옆 상자 「풀타임 환산 WAR로 정렬하면」 (SA-6 aside) */
function fullTimeBox(): HTMLElement {
  const name = /풀타임 환산 WAR로 정렬하면/;
  const found = [...screen.queryAllByRole("complementary", { name }), ...screen.queryAllByRole("region", { name })];
  expect(found.length, '옆 상자 "풀타임 환산 WAR로 정렬하면" (aside · region, 이름 있음)').toBeGreaterThan(0);
  return found[0]!;
}

const countLine = (n: number): string => `2026 출전 선수 ${fmtCount(n)}명`;

/** 그룹 평균 오차 문장 (D39) — 숫자는 warGroupMae(evaluations, 역할, "seasons_3_5")를 fmtWar로 */
function groupMaeSentence(file: WarFile): string {
  const bat = fmtWar(warGroupMae(file.evaluations, "bat", "seasons_3_5"));
  const pit = fmtWar(warGroupMae(file.evaluations, "pit", "seasons_3_5"));
  return `비슷한 경력 선수의 예측 WAR은 과거 검증에서 평균 ${bat}승(3~5시즌 타자)·${pit}승(투수) 빗나갔습니다.`;
}

// ---------------------------------------------------------------------------
// 테스트
// ---------------------------------------------------------------------------

describe("WAR 순위 탭 — 자리와 지연 로딩 (SA-6, N6)", () => {
  it("예측표 탭이 셋이다 — 타자 · 투수 · WAR 순위, 기본은 타자", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    expect(isSelected(roleTab("타자"))).toBe(true);
    expect(isSelected(roleTab("투수"))).toBe(false);
    expect(isSelected(roleTab("WAR 순위"))).toBe(false);
    // 타자 탭에는 WAR 전용 정렬 세그먼트가 없다
    expect(screen.queryAllByRole("radio", { name: SEG_FULL })).toHaveLength(0);
  });

  it("첫 화면(타자 탭)과 투수 탭에서는 /data/war.json을 받지 않는다", async () => {
    const fetchMock = stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    // 다른 요청이 다 끝날 때까지 기다린 뒤에도 war.json은 0번
    await waitFor(() => expect(callsTo(fetchMock, "/data/batters.json")).toBe(1));
    expect(callsTo(fetchMock, "/data/war.json")).toBe(0);
    fireEvent.click(roleTab("투수"));
    await waitFor(() => expect(callsTo(fetchMock, "/data/pitchers.json")).toBe(1));
    expect(callsTo(fetchMock, "/data/war.json")).toBe(0);
  });

  it("WAR 탭을 처음 누를 때 war.json을 한 번만 받는다 — 다른 탭에 갔다 와도 다시 받지 않는다", async () => {
    const fetchMock = stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    await openWarTab();
    expect(isSelected(roleTab("WAR 순위"))).toBe(true);
    expect(isSelected(roleTab("타자"))).toBe(false);
    expect(callsTo(fetchMock, "/data/war.json")).toBe(1);

    fireEvent.click(roleTab("타자"));
    await waitFor(() => expect(rowIds(getTable())).toEqual(BAT.rows.filter((r) => r.prev !== null).map((r) => r.player.player_id)));
    await openWarTab();
    expect(callsTo(fetchMock, "/data/war.json")).toBe(1);
    expect(callsTo(fetchMock, "/data/batters.json")).toBe(1);
  });

  it("주소 해시가 role=war이면 그 탭으로 열고 war.json을 받는다", async () => {
    arriveAt("/#role=war");
    const fetchMock = stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await waitFor(() => expect(rowIds(getTable())).toEqual(WAR_ORDER), { timeout: 3000 });
    expect(isSelected(roleTab("WAR 순위"))).toBe(true);
    expect(callsTo(fetchMock, "/data/war.json")).toBe(1);
  });
});

describe("WAR 순위표 — 열과 칸 (SA-6, F21, D34)", () => {
  it("열: # · 선수 · 역할 · 2027 예측 WAR* · 풀타임 환산 · 2026 WAR · 예측 출전*", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    const table = await openWarTab();
    const expected = [COL.rank, COL.player, COL.role, COL.war, COL.full, COL.prev, COL.pt];
    const texts = headers(table).map((h) => norm(h.textContent));
    expect(texts).toHaveLength(expected.length);
    expected.forEach((re, i) => expect(texts[i]).toMatch(re));
  });

  it("기본 정렬은 예측 WAR 높은 순 — 같으면 이름순, 순위 번호는 1부터", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    const table = await openWarTab();
    expect(isSelected(sortSegment(SEG_WAR))).toBe(true);
    expect(isSelected(sortSegment(SEG_FULL))).toBe(false);
    WAR_ORDER.forEach((id, i) => expect(cellText(table, id, COL.rank)).toBe(String(i + 1)));
  });

  it("투타 겸업 한 행: 사진·이름(상세 링크)·포지션·팀, 투타 배지, 큰 합계 아래 '타 + 투', 풀타임 환산, 2026 WAR, '500타석 · 77이닝'", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    const table = await openWarTab();
    const row = warOf(302);

    const player = cellOf(table, rowOf(table, 302), COL.player);
    const links = within(player).getAllByRole("link");
    expect(links.some((a) => norm(a.textContent).includes("Two Way Star"))).toBe(true);
    for (const a of within(rowOf(table, 302)).getAllByRole("link")) expectHref(a, playerHref(302));
    expect(norm(player.textContent)).toMatch(/TWP\s*·\s*Sample Coast/);

    expect(cellText(table, 302, COL.role)).toBe(warRoleLabel("both")); // 투타
    const war = cellText(table, 302, COL.war);
    expect(war).toContain(fmtWar(row.war)); // 6.2
    expect(war).toContain(fmtWarSplit(row.war_bat, row.war_pit)); // 타 3.9 + 투 2.3
    expect(cellText(table, 302, COL.full)).toBe(fmtWar(row.war_full)); // 10.0
    expect(cellText(table, 302, COL.prev)).toBe(fmtWar(row.prev_war)); // 6.6
    expect(cellText(table, 302, COL.pt)).toMatch(new RegExp(`^${playingTimePattern(row).source}$`)); // 500타석 · 77이닝
  });

  it("타자만 · 투수만인 행 — 배지 타 / 투, 분해 줄 없음, 출전은 타석 또는 이닝 하나", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    const table = await openWarTab();

    const bat = warOf(301);
    expect(cellText(table, 301, COL.role)).toBe(warRoleLabel("bat")); // 타
    expect(cellText(table, 301, COL.war)).toBe(fmtWar(bat.war)); // 6.6 — "타 … + 투 …" 없음
    expect(cellText(table, 301, COL.full)).toBe(fmtWar(bat.war_full));
    expect(cellText(table, 301, COL.prev)).toBe(fmtWar(bat.prev_war)); // 10.6
    expect(cellText(table, 301, COL.pt)).toBe(`${fmtCount(bat.pa)}타석`); // 594타석
    expect(norm(cellOf(table, rowOf(table, 301), COL.player).textContent)).toMatch(/CF\s*·\s*Sample Lakes/);

    const pit = warOf(304);
    expect(cellText(table, 304, COL.role)).toBe(warRoleLabel("pit")); // 투
    expect(cellText(table, 304, COL.war)).toBe(fmtWar(pit.war)); // 4.3
    expect(cellText(table, 304, COL.pt)).toBe(`${fmtCount(pit.ip)}이닝`); // 132이닝
    expect(cellText(table, 304, COL.pt)).not.toMatch(/타석/);
  });

  it("2026 WAR이 없으면 —, 음수는 −, 포지션·팀이 비어도 null·undefined·NaN을 쓰지 않는다", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    const table = await openWarTab();
    expect(cellText(table, 305, COL.prev)).toBe(DASH);
    const neg = warOf(308);
    expect(cellText(table, 308, COL.war)).toBe(fmtWar(neg.war)); // −0.3
    expect(cellText(table, 308, COL.full)).toBe(fmtWar(neg.war_full)); // −0.4
    expect(cellText(table, 308, COL.prev)).toBe(fmtWar(neg.prev_war));
    expect(fmtWar(neg.war).startsWith("−")).toBe(true);
    for (const id of WAR_ORDER) expect(norm(rowOf(table, id).textContent)).not.toMatch(/null|undefined|NaN/);
  });

  it("선수 사진 — MLB 서버 사진(playerPhotoUrl), 못 불러오면 이니셜", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    const table = await openWarTab();
    const photos = mlbPhotos(cellOf(table, rowOf(table, 301), COL.player));
    expect(photos).toHaveLength(1);
    const anyWidth = (url: string) => url.replace(/w_\d+/, "w_N");
    expect(anyWidth(photos[0]!.getAttribute("src")!)).toBe(anyWidth(playerPhotoUrl(301)));
    fireEvent.error(photos[0]!);
    await waitFor(() => expect(mlbPhotos(cellOf(getTable(), rowOf(getTable(), 301), COL.player))).toHaveLength(0));
    expect(showsInitials(cellOf(getTable(), rowOf(getTable(), 301), COL.player), initials("Alpha Center"))).toBe(true);
  });

  it("사진을 끄면(NEXT_PUBLIC_PLAYER_PHOTOS=off) 처음부터 이니셜만", async () => {
    vi.stubEnv("NEXT_PUBLIC_PLAYER_PHOTOS", "off");
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    const table = await openWarTab();
    expect(mlbPhotos(document.body)).toHaveLength(0);
    expect(showsInitials(cellOf(table, rowOf(table, 302), COL.player), initials("Two Way Star"))).toBe(true); // "TW"
  });
});

describe("WAR 순위 — 정렬 · 검색 · 대상 수 (SA-6, D35 · D42)", () => {
  it("「풀타임 환산 WAR」을 고르면 war_full 높은 순(같으면 이름순)으로 바뀌고 순위 번호도 새로 매긴다", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    await openWarTab();
    fireEvent.click(sortSegment(SEG_FULL));
    await waitFor(() => expect(rowIds(getTable())).toEqual(FULL_ORDER));
    expect(isSelected(sortSegment(SEG_FULL))).toBe(true);
    expect(isSelected(sortSegment(SEG_WAR))).toBe(false);
    const table = getTable();
    FULL_ORDER.forEach((id, i) => expect(cellText(table, id, COL.rank)).toBe(String(i + 1)));

    fireEvent.click(sortSegment(SEG_WAR));
    await waitFor(() => expect(rowIds(getTable())).toEqual(WAR_ORDER));
    expect(cellText(getTable(), 301, COL.rank)).toBe("1");
  });

  it("이름으로 찾는다 — 대소문자·악센트를 가리지 않는다 (기존 예측표 검색과 같은 방식)", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    await openWarTab();
    search("tie breaker");
    await waitFor(() => expect(rowIds(getTable())).toEqual([307, 306]));
    search("JOSE");
    await waitFor(() => expect(rowIds(getTable())).toEqual([309]));
    search("");
    await waitFor(() => expect(rowIds(getTable())).toEqual(WAR_ORDER));
  });

  it("없는 선수 — 예측표에는 있지만 2026에 뛰지 않은 선수를 찾으면 빈 안내와 'WAR 순위는 2026 출전 선수만'", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    await openWarTab();
    search("Bench Only"); // BAT에는 있고 war.json에는 없다 (D42)
    await waitFor(() => expect(screen.queryByRole("table") === null || rowIds(getTable()).length === 0).toBe(true));
    expect(screen.getAllByText(/없습니다/).length).toBeGreaterThan(0);
    expect(blocksWith(WAR_PREV_ONLY).length, WAR_PREV_ONLY).toBeGreaterThan(0);
    // 빈 안내의 단추(「필터 지우기」 또는 「검색 지우기」)로 검색을 지우면 전원이 돌아온다.
    // 검색 칸의 × 단추("선수 이름 검색 지우기")와 헷갈리지 않게 앞을 고정한다
    fireEvent.click(screen.getByRole("button", { name: /^(필터|검색) 지우기/ }));
    await waitFor(() => expect(rowIds(getTable())).toEqual(WAR_ORDER));
    expect(searchBox().value).toBe("");
  });

  it("체크 대신 '2026 출전 선수 N명'을 적는다 — 「2026 출전 선수만」·「예상 100이닝 이상」 토글은 없다 (D42)", async () => {
    const BIG = bigWar(1470);
    stubData({ [BAT_URL]: BAT, [WAR_URL]: BIG });
    renderHome();
    await findTable();
    await openWarTab(orderBy(BIG, "war").slice(0, PAGE_SIZE));
    expect(blocksWith(countLine(1470)).length, countLine(1470)).toBeGreaterThan(0); // "2026 출전 선수 1,470명"
    for (const name of [/2026 출전/, /100이닝/]) {
      expect(screen.queryAllByRole("switch", { name })).toHaveLength(0);
      expect(screen.queryAllByRole("checkbox", { name })).toHaveLength(0);
    }
  }, 20000);
});

describe("WAR 순위 — 옆 상자와 표 아래 설명 (SA-6, D33 · D39 · D41)", () => {
  it("옆 상자 「풀타임 환산 WAR로 정렬하면」 — war_full 상위 5명, 각 '예측 X.X · N타석(또는 N이닝)'", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    await openWarTab();
    const box = fullTimeBox();
    const items = within(box).getAllByRole("listitem");
    expect(items).toHaveLength(5);
    FULL_ORDER.slice(0, 5).forEach((id, i) => {
      const row = warOf(id);
      const text = norm(items[i]!.textContent);
      expect(text).toContain(row.player.name);
      expect(text).toContain(fmtWar(row.war_full));
      expect(text).toMatch(new RegExp(`예측\\s*${fmtWar(row.war).replace(".", "\\.")}\\s*·\\s*${playingTimePattern(row).source}`));
    });
    // 출전 짧은 선수(303)는 "예측 2.8 · 252타석"
    expect(norm(items[2]!.textContent)).toMatch(/예측\s*2\.8\s*·\s*252\s*타석/);
    // 정렬을 바꿔도 상자는 그대로 (전원 기준 상위 5)
    fireEvent.click(sortSegment(SEG_FULL));
    await waitFor(() => expect(rowIds(getTable())).toEqual(FULL_ORDER));
    expect(within(fullTimeBox()).getAllByRole("listitem")).toHaveLength(5);
  });

  it("표 아래: WAR 설명 · 보수적 예상 · 그룹 평균 오차(3~5시즌 타자 · 투수) · 풀타임 기준", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    const table = await openWarTab();
    const sentence = groupMaeSentence(WAR);
    // 평균을 쓴다 — 한 시즌 값(0.7 · 0.5)이나 다른 그룹(전체 0.9)이 아니다
    expect(sentence).toBe("비슷한 경력 선수의 예측 WAR은 과거 검증에서 평균 0.6승(3~5시즌 타자)·0.4승(투수) 빗나갔습니다.");
    // 기준 출전량은 옆 상자 문장(「…65이닝을 뛴다면」, SA-6)에 있다
    for (const note of [WAR_ABOUT, WAR_CONSERVATIVE, sentence, fullTimeUnits(WAR.full_time), FULL_TIME_ADJUSTED]) {
      const found = blocksWith(note);
      expect(found.length, note).toBeGreaterThan(0);
      expect(found.some((el) => follows(table, el)), `표 뒤에: ${note}`).toBe(true);
    }
    expect(norm(document.body.textContent)).toContain("WAR(MLB Stats API 기준)");
  });

  it("WAR 탭에는 ± 범위 설명(BAND_NOTE)·PA·HR 출전량 설명을 달지 않는다 (D39) — 타자 탭으로 돌아가면 다시 보인다", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    await openWarTab();
    expect(blocksWith(BAND_NOTE)).toHaveLength(0);
    expect(blocksWith(PLAYING_TIME_NOTE)).toHaveLength(0);
    fireEvent.click(roleTab("타자"));
    await waitFor(() => expect(blocksWith(BAND_NOTE).length).toBeGreaterThan(0));
  });

  it("표 뒤에 검증 화면 링크(/validation/)가 있다 — 「WAR은 어떻게 계산되나」", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    const table = await openWarTab();
    const links = screen
      .getAllByRole("link")
      .filter((a) => /^\/validation\/?(#.*)?$/.test(a.getAttribute("href") ?? "") && follows(table, a));
    expect(links.length).toBeGreaterThan(0);
  });

  it("제목 단계가 건너뛰지 않는다 — WAR 탭에서도 h1 하나로 시작 (N10)", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    await openWarTab();
    const levels = screen
      .getAllByRole("heading")
      .map((h) => Number(h.getAttribute("aria-level") ?? h.tagName.replace(/^H/, "")));
    expect(levels[0]).toBe(1);
    expect(levels.filter((l) => l === 1)).toHaveLength(1);
    levels.slice(1).forEach((level, i) => {
      expect(level, `제목 단계 ${levels.join(",")}`).toBeLessThanOrEqual(levels[i]! + 1);
    });
  });
});

describe(`WAR 순위 — 페이지 나눔 (한 쪽 ${PAGE_SIZE}행)`, () => {
  const BIG = bigWar(1470);
  const byWar = orderBy(BIG, "war");
  const byFull = orderBy(BIG, "war_full");

  it("50행씩 보이고 2쪽으로 넘어가면 순위 번호가 51부터", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: BIG });
    renderHome();
    await findTable();
    await openWarTab(byWar.slice(0, PAGE_SIZE));
    goToPage(2);
    await waitFor(() => expect(rowIds(getTable())).toEqual(byWar.slice(PAGE_SIZE, PAGE_SIZE * 2)));
    expect(cellText(getTable(), byWar[PAGE_SIZE]!, COL.rank)).toBe(String(PAGE_SIZE + 1));
    expect(cellText(getTable(), byWar[PAGE_SIZE * 2 - 1]!, COL.rank)).toBe(String(PAGE_SIZE * 2));
  }, 20000);

  it("정렬·검색을 바꾸면 1쪽으로 돌아간다", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: BIG });
    renderHome();
    await findTable();
    await openWarTab(byWar.slice(0, PAGE_SIZE));
    goToPage(2);
    await waitFor(() => expect(rowIds(getTable())).toEqual(byWar.slice(PAGE_SIZE, PAGE_SIZE * 2)));
    fireEvent.click(sortSegment(SEG_FULL));
    await waitFor(() => expect(rowIds(getTable())).toEqual(byFull.slice(0, PAGE_SIZE)));
    expect(cellText(getTable(), byFull[0]!, COL.rank)).toBe("1");

    goToPage(2);
    await waitFor(() => expect(rowIds(getTable())).toEqual(byFull.slice(PAGE_SIZE, PAGE_SIZE * 2)));
    search("War Player"); // 모두 맞는다 — 결과는 같고 쪽만 1로
    await waitFor(() => expect(rowIds(getTable())).toEqual(byFull.slice(0, PAGE_SIZE)));
  }, 30000);
});

describe("WAR 순위 — 주소 해시 (DESIGN §1-2, 선수 상세에서 뒤로)", () => {
  it("WAR 탭은 role=war, 풀타임 정렬은 wsort=full, 기본 정렬(예측 WAR)은 적지 않는다", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await findTable();
    await openWarTab();
    const hashParams = () => new URLSearchParams(window.location.hash.slice(1));
    await waitFor(() => expect(hashParams().get("role")).toBe("war"));
    expect(hashParams().has("wsort")).toBe(false);
    fireEvent.click(sortSegment(SEG_FULL));
    await waitFor(() => expect(hashParams().get("wsort")).toBe("full"));
    search("tie");
    await waitFor(() => expect(hashParams().get("q")).toBe("tie"));
    fireEvent.click(sortSegment(SEG_WAR));
    await waitFor(() => expect(hashParams().has("wsort")).toBe(false));
    expect(hashParams().get("role")).toBe("war");
  });

  it("해시로 돌아오면 WAR 탭·풀타임 정렬·쪽을 되살린다 — 순위 번호도 그 쪽 그대로", async () => {
    const BIG = bigWar(1470);
    const byFull = orderBy(BIG, "war_full");
    arriveAt("/#role=war&wsort=full&page=2");
    const fetchMock = stubData({ [BAT_URL]: BAT, [WAR_URL]: BIG });
    renderHome();
    await waitFor(() => expect(rowIds(getTable())).toEqual(byFull.slice(PAGE_SIZE, PAGE_SIZE * 2)), { timeout: 3000 });
    expect(isSelected(roleTab("WAR 순위"))).toBe(true);
    expect(isSelected(sortSegment(SEG_FULL))).toBe(true);
    expect(cellText(getTable(), byFull[PAGE_SIZE]!, COL.rank)).toBe(String(PAGE_SIZE + 1));
    expect(callsTo(fetchMock, "/data/war.json")).toBe(1);
  }, 20000);

  it("해시의 검색어로 돌아온다 — 검색 칸과 결과가 그대로", async () => {
    arriveAt("/#role=war&q=tie");
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await waitFor(() => expect(rowIds(getTable())).toEqual([307, 306]), { timeout: 3000 });
    expect(searchBox().value).toBe("tie");
    expect(isSelected(sortSegment(SEG_WAR))).toBe(true);
  });

  it("viewState — role=war와 wsort=full을 읽고 쓰며, 틀린 wsort는 기본값, 기존 role=pit 해석은 그대로", () => {
    expect(isViewHash("#wsort=full")).toBe(true);
    expect(decodeView("#role=war")?.role).toBe("war");
    expect(decodeView("#role=pit")?.role).toBe("pit");
    expect(decodeView("#role=xx")?.role).toBe("bat");

    const back = decodeView("#role=war&wsort=full&q=Jos%C3%A9&page=3");
    expect(back).not.toBeNull();
    const params = new URLSearchParams(encodeView(back!));
    expect(params.get("role")).toBe("war");
    expect(params.get("wsort")).toBe("full");
    expect(params.get("q")).toBe("José");
    expect(params.get("page")).toBe("3");
    // 왕복
    expect(decodeView(`#${encodeView(back!)}`)).toEqual(back);

    // 기본 정렬(예측 WAR)과 틀린 값은 해시에 남기지 않는다
    expect(encodeView(decodeView("#role=war")!)).toBe("role=war");
    expect(encodeView(decodeView("#role=war&wsort=bogus")!)).toBe("role=war");
  });
});

describe("WAR 순위 — 모바일 (SA-9, 390px)", () => {
  it("표 대신 선수 카드 — 순위·이름·포지션·팀·역할 배지, 2027 WAR*(투타는 분해 한 줄)·풀타임 환산만", async () => {
    setViewport(390);
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await cardIds(); // 타자 탭 카드
    fireEvent.click(roleTab("WAR 순위"));
    await waitFor(async () => expect(await cardIds()).toEqual(WAR_ORDER), { timeout: 3000 });
    expect(screen.queryByRole("table")).toBeNull();

    const card = cards()[1]!; // 2위 투타 겸업
    expect(rowIdOf(card)).toBe(302);
    const row = warOf(302);
    for (const a of within(card).getAllByRole("link")) expectHref(a, playerHref(302));
    const text = norm(card.textContent);
    expect(within(card).getAllByText(/^2$/).length, "순위 번호 2").toBeGreaterThan(0);
    expect(text).toContain("Two Way Star");
    expect(text).toMatch(/TWP\s*·\s*Sample Coast/);
    expect(text).toContain(warRoleLabel("both"));
    expect(text).toMatch(/2027 WAR\*/);
    expect(text).toContain(fmtWar(row.war)); // 6.2
    expect(text).toContain(fmtWarSplit(row.war_bat, row.war_pit)); // 타 3.9 + 투 2.3
    expect(text).toMatch(/풀타임 환산/);
    expect(text).toContain(fmtWar(row.war_full)); // 10.0
    // 2026 WAR · 예측 출전은 카드에 두지 않는다 (선수 상세에서)
    expect(text).not.toMatch(/2026 WAR/);
    expect(text).not.toContain(fmtWar(row.prev_war)); // 6.6
    expect(text).not.toMatch(/타석|이닝/);

    // 타자만인 카드에는 분해 줄이 없다
    expect(norm(cards()[0]!.textContent)).not.toMatch(/타 [−\d.]+ \+ 투/);
  });

  it("50명씩 보이고 '50명 더 보기'로 이어 붙인다 — 쪽 단추는 없다", async () => {
    setViewport(390);
    const BIG = bigWar(1470);
    const byWar = orderBy(BIG, "war");
    stubData({ [BAT_URL]: BAT, [WAR_URL]: BIG });
    renderHome();
    await cardIds();
    fireEvent.click(roleTab("WAR 순위"));
    await waitFor(async () => expect(await cardIds()).toEqual(byWar.slice(0, PAGE_SIZE)), { timeout: 5000 });
    expect(screen.queryAllByRole("navigation", { name: /페이지/ })).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: `${fmtCount(PAGE_SIZE)}명 더 보기` }));
    await waitFor(async () => expect(await cardIds()).toEqual(byWar.slice(0, PAGE_SIZE * 2)));
  }, 20000);

  it("정렬 세그먼트 「풀타임 환산」으로 카드 순서가 바뀐다", async () => {
    setViewport(390);
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await cardIds();
    fireEvent.click(roleTab("WAR 순위"));
    await waitFor(async () => expect(await cardIds()).toEqual(WAR_ORDER), { timeout: 3000 });
    fireEvent.click(sortSegment(SEG_FULL));
    await waitFor(async () => expect(await cardIds()).toEqual(FULL_ORDER));
  });

  it("대상 수 · WAR 표기 · 각주 '2026 WAR과 예측 출전은 선수 상세에서'", async () => {
    setViewport(390);
    stubData({ [BAT_URL]: BAT, [WAR_URL]: WAR });
    renderHome();
    await cardIds();
    fireEvent.click(roleTab("WAR 순위"));
    await waitFor(async () => expect(await cardIds()).toEqual(WAR_ORDER), { timeout: 3000 });
    expect(blocksWith(countLine(WAR.rows.length)).length).toBeGreaterThan(0); // 2026 출전 선수 9명
    expect(blocksWith("2026 WAR과 예측 출전은 선수 상세에서").length).toBeGreaterThan(0);
    expect(norm(document.body.textContent)).toContain("WAR(MLB Stats API 기준)");
  });

  it("좁은 화면 해시 page=2로 돌아오면 100명까지 보인다", async () => {
    setViewport(390);
    const BIG = bigWar(1470);
    arriveAt("/#role=war&page=2");
    stubData({ [BAT_URL]: BAT, [WAR_URL]: BIG });
    renderHome();
    await waitFor(async () => expect(await cardIds()).toEqual(orderBy(BIG, "war").slice(0, PAGE_SIZE * 2)), {
      timeout: 5000,
    });
  }, 20000);
});

describe("WAR 순위 — 불러오기 상태 (기존 LoadingPanel · ErrorPanel 패턴)", () => {
  it("받는 동안 'WAR 순위를 불러오는 중'을 보이고, 다 받으면 표로 바꾼다", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    stubData({
      [BAT_URL]: BAT,
      [WAR_URL]: async () => {
        await gate;
        return jsonResponse(WAR);
      },
    });
    renderHome();
    await findTable();
    fireEvent.click(roleTab("WAR 순위"));
    await waitFor(() => expect(screen.getAllByText(/WAR 순위를 불러오는 중/).length).toBeGreaterThan(0));
    expect(document.querySelector('[aria-busy="true"]')).not.toBeNull();
    expect(screen.queryByRole("table")).toBeNull();
    release();
    await waitFor(() => expect(rowIds(getTable())).toEqual(WAR_ORDER), { timeout: 3000 });
    await waitFor(() => expect(screen.queryAllByText(/불러오는 중/)).toHaveLength(0));
  });

  it("받지 못하면(HTTP 500) 'WAR 순위를 불러오지 못했습니다' — '다시 시도'로 다시 받는다", async () => {
    let calls = 0;
    const fetchMock = stubData({
      [BAT_URL]: BAT,
      [WAR_URL]: async () => (calls++ === 0 ? new Response("boom", { status: 500 }) : jsonResponse(WAR)),
    });
    renderHome();
    await findTable();
    fireEvent.click(roleTab("WAR 순위"));
    await waitFor(() => expect(warErrorShown()).toBe(true), { timeout: 3000 });
    expect(screen.queryByRole("table")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /다시 시도/ }));
    await waitFor(() => expect(rowIds(getTable())).toEqual(WAR_ORDER), { timeout: 3000 });
    expect(callsTo(fetchMock, "/data/war.json")).toBe(2);
    expect(warErrorShown()).toBe(false);
  });

  it("모양이 틀린 war.json(출처가 다름)도 오류 안내 — 타자 탭은 그대로", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: { ...WAR, source: "FanGraphs" } });
    renderHome();
    await findTable();
    fireEvent.click(roleTab("WAR 순위"));
    await waitFor(() => expect(warErrorShown()).toBe(true), { timeout: 3000 });
    fireEvent.click(roleTab("타자"));
    await waitFor(() => expect(rowIds(getTable()).length).toBeGreaterThan(0));
  });

  it("war.json이 없으면(404) 오류 안내 — 머리글·바닥글은 그대로", async () => {
    stubData({ [BAT_URL]: BAT }); // war.json → 404
    renderHome();
    await findTable();
    fireEvent.click(roleTab("WAR 순위"));
    await waitFor(() => expect(warErrorShown()).toBe(true), { timeout: 3000 });
    expect(screen.getByRole("navigation", { name: "주요 메뉴" })).toBeInTheDocument();
    expect(document.querySelector("footer")).toHaveTextContent("MLB Stats API, Baseball Savant");
  });
});

describe("WAR 순위 — 빈 데이터 (Refactor 리뷰에서 더함)", () => {
  it("과거 검증 행이 없으면 그룹 평균 오차 문장을 쓰지 않는다 — '평균 —승'이 보이지 않는다", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: warFile(WAR.rows, []) });
    renderHome();
    await findTable();
    await openWarTab();
    const body = norm(document.body.textContent);
    expect(body).toContain(WAR_CONSERVATIVE);
    expect(body).not.toContain("과거 검증에서 평균");
    expect(body).not.toContain(`${DASH}승`);
  });

  it("투수 검증 행만 없으면 타자 오차만 적는다", async () => {
    const batOnly = EVALS.filter((e) => e.role === "bat");
    stubData({ [BAT_URL]: BAT, [WAR_URL]: warFile(WAR.rows, batOnly) });
    renderHome();
    await findTable();
    await openWarTab();
    const body = norm(document.body.textContent);
    expect(body).toContain("과거 검증에서 평균 0.6승(3~5시즌 타자) 빗나갔습니다.");
    expect(body).not.toContain(`${DASH}승`);
  });

  it("war.json에 선수가 하나도 없으면 빈 안내 — 지울 검색이 없으니 「검색 지우기」 단추도 없다", async () => {
    stubData({ [BAT_URL]: BAT, [WAR_URL]: warFile([]) });
    renderHome();
    await findTable();
    fireEvent.click(roleTab("WAR 순위"));
    expect(await screen.findByText("WAR 순위에 실을 선수가 없습니다", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByText(WAR_PREV_ONLY)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "검색 지우기" })).toBeNull();
    expect(norm(document.body.textContent)).toContain(countLine(0));
  });
});
