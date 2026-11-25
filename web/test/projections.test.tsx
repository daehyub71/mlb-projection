// 예측표 (F13, DESIGN SA-1 데스크톱 · SA-3 모바일) — 레인 A Red 테스트
// 데이터는 가짜 fetch로 준다(네트워크 없음). 숫자 기대값은 모두 src/format.ts 함수로 만든다.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import HomePage from "@/app/page";
import { Providers } from "@/app/providers";
import { DATA_BASE, parsePlayerId } from "@/src/data/load";
import type { BatRow, BatTable, Meta, PitRow, PitTable } from "@/src/data/types";
import { DASH, fmtBand, fmtCount, fmtDiff, fmtEra, fmtIp, fmtPct, fmtRate } from "@/src/format";
import { initials, playerPhotoUrl } from "@/src/photo";
import { BAND_NOTE, PAGE_SIZE, PLAYING_TIME_NOTE, playerHref } from "@/src/site";
import { filterRows, nextSort, normalizeName, pageCount, sortRows, toBatViews } from "@/app/projections/rows";
import { DEFAULT_VIEW, decodeView, encodeView, isViewHash } from "@/app/projections/viewState";

// ---------------------------------------------------------------------------
// 픽스처 — 합성 샘플의 meta를 바탕으로 손으로 만든 표
// ---------------------------------------------------------------------------

const ROOT = join(import.meta.dirname, "..", "..");
const sampleBat = JSON.parse(
  readFileSync(join(ROOT, "tests/fixtures/synthetic/web/batters.json"), "utf-8"),
) as BatTable;
const sampleMeta: Meta = {
  ...sampleBat.meta,
  disclaimer:
    "재미로 보는 예측입니다. 최근 5시즌 기록으로 만든 통계 모델의 점 예측이며, 실제 성적을 보장하지 않습니다.",
};

const LEAGUE_OPS = 0.719;

interface BatSpec {
  id: number;
  name: string;
  pos?: string | null;
  age?: number | null;
  team?: string | null;
  active?: boolean;
  last?: number;
  n: number;
  sample: number;
  pa: number;
  hr: number;
  ops: number;
  prev: { pa: number; hr: number; ops: number | null } | null;
}

function batRow(s: BatSpec): BatRow {
  return {
    player: {
      player_id: s.id,
      name: s.name,
      primary_position: s.pos === undefined ? "1B" : s.pos,
      age: s.age === undefined ? 28 : s.age,
      team_name: s.team === undefined ? "Sample Team" : s.team,
      active: s.active ?? true,
      last_mlb_season: s.last ?? (s.prev === null ? 2025 : 2026),
    },
    window: { input_first: 2027 - s.n, input_last: 2026, n_seasons: s.n },
    sample_pa: s.sample,
    projection: { pa: s.pa, hr: s.hr, obp: 0.33, slg: 0.45, ops: s.ops },
    prev: s.prev === null ? null : { season: 2026, ...s.prev },
  };
}

function batTable(rows: BatRow[]): BatTable {
  return {
    meta: sampleMeta,
    role: "bat",
    model: "extended",
    league_baseline: { ops: LEAGUE_OPS },
    bands: { seasons_1_2: { ops: 0.104 }, seasons_3_5: { ops: 0.096 } },
    rows,
  };
}

// JSON 순서는 일부러 섞었다 — 화면이 직접 정렬해야 한다
const BAT_SPECS: BatSpec[] = [
  { id: 101, name: "Pete Alonso", pos: "1B", age: 32, team: "Baltimore Orioles", n: 5, sample: 3447, pa: 558.4, hr: 30.2, ops: 0.842, prev: { pa: 690, hr: 38, ops: 0.885 } },
  { id: 102, name: "Yordan Alvarez", pos: "DH", age: 30, team: "Houston Astros", n: 5, sample: 2583, pa: 504.2, hr: 27.4, ops: 0.939, prev: { pa: 600, hr: 33, ops: 1.033 } },
  { id: 103, name: "James Wood", pos: "RF", age: 24, team: "Washington Nationals", n: 3, sample: 1648, pa: 544.0, hr: 26.1, ops: 0.845, prev: { pa: 650, hr: 31, ops: 0.891 } },
  { id: 104, name: "Bobby Witt Jr.", pos: "SS", age: 27, team: "Kansas City Royals", n: 5, sample: 3354, pa: 506.3, hr: 17.2, ops: 0.835, prev: { pa: 700, hr: 22, ops: 0.8 } },
  { id: 105, name: "Rookie Twoyear", pos: "2B", age: 23, n: 2, sample: 420, pa: 380.6, hr: 9.4, ops: 0.702, prev: { pa: 300, hr: 6, ops: 0.69 } },
  { id: 106, name: "Retired Slugger", pos: "LF", age: 38, team: "Old Club", active: false, last: 2024, n: 3, sample: 1500, pa: 120.2, hr: 4.1, ops: 0.76, prev: null },
  { id: 107, name: "Bench Bat", pos: null, age: null, team: null, n: 1, sample: 45, pa: 60.3, hr: 1.2, ops: 0.65, prev: null },
  { id: 108, name: "Zero Ops", pos: "C", age: 31, n: 4, sample: 900, pa: 50.1, hr: 0.6, ops: 0.68, prev: { pa: 3, hr: 0, ops: null } },
];
const BAT = batTable(BAT_SPECS.map(batRow));
const batSpec = (id: number): BatSpec => BAT_SPECS.find((s) => s.id === id)!;

// 기본 보기(2026 출전만, OPS 높은 순)와 전원 보기
const BAT_DEFAULT = [102, 103, 101, 104, 105, 108];
const BAT_ALL = [102, 103, 101, 104, 106, 105, 108, 107];

interface PitSpec {
  id: number;
  name: string;
  pos?: string;
  age?: number;
  team?: string | null;
  active?: boolean;
  n: number;
  bf: number;
  sampleIp: number;
  ip: number;
  era: number;
  kbb: number;
  prev: { ip: number; era: number | null; kbb: number | null } | null;
}

function pitRow(s: PitSpec): PitRow {
  return {
    player: {
      player_id: s.id,
      name: s.name,
      primary_position: s.pos ?? "P",
      age: s.age ?? 29,
      team_name: s.team === undefined ? "Sample Team" : s.team,
      active: s.active ?? true,
      last_mlb_season: s.prev === null ? 2025 : 2026,
    },
    window: { input_first: 2027 - s.n, input_last: 2026, n_seasons: s.n },
    sample_bf: s.bf,
    sample_ip: s.sampleIp,
    projection: { ip: s.ip, era: s.era, k_bb_pct: s.kbb },
    prev: s.prev === null ? null : { season: 2026, ip: s.prev.ip, era: s.prev.era, k_bb_pct: s.prev.kbb },
  };
}

function pitTable(rows: PitRow[]): PitTable {
  return {
    meta: sampleMeta,
    role: "pit",
    model: "extended",
    league_baseline: { era: 4.15, k_bb_pct: 13.3 },
    bands: { seasons_1_2: { era: 1.49, k_bb_pct: 5.9 }, seasons_3_5: { era: 1.38, k_bb_pct: 5.5 } },
    rows,
  };
}

const PIT_SPECS: PitSpec[] = [
  { id: 201, name: "Tarik Skubal", age: 30, team: "Detroit Tigers", n: 5, bf: 3800, sampleIp: 950.3, ip: 172.3333, era: 2.82, kbb: 22.4, prev: { ip: 190, era: 2.45, kbb: 24.1 } },
  { id: 202, name: "Paul Skenes", age: 25, team: "Pittsburgh Pirates", n: 3, bf: 1900, sampleIp: 470, ip: 160.7, era: 2.75, kbb: 19.8, prev: { ip: 180.3333, era: 2.2, kbb: 21 } },
  { id: 203, name: "Middle Reliever", age: 29, n: 2, bf: 400, sampleIp: 95.7, ip: 41.3, era: 4.05, kbb: 14.2, prev: { ip: 50, era: 4.5, kbb: 12 } },
  { id: 204, name: "Shohei Ohtani", pos: "TWP", age: 32, team: "Los Angeles Dodgers", n: 4, bf: 1500, sampleIp: 380, ip: 120, era: 3.1, kbb: 20.5, prev: { ip: 80, era: 2.87, kbb: 18.4 } },
  { id: 205, name: "Gone Arm", age: 33, team: null, active: false, n: 2, bf: 300, sampleIp: 70, ip: 20, era: 5.2, kbb: 6.1, prev: null },
];
const PIT = pitTable(PIT_SPECS.map(pitRow));
const pitSpec = (id: number): PitSpec => PIT_SPECS.find((s) => s.id === id)!;
const PIT_DEFAULT = [202, 201, 204, 203]; // 2026 출전만, ERA 낮은 순

/** 한 쪽(50행)을 넘는 큰 타자 표 — 10번째마다 2026 미출전, 사용 시즌은 2와 5를 번갈아 */
function bigBatters(count: number): BatTable {
  return batTable(
    Array.from({ length: count }, (_, i) =>
      batRow({
        id: 1001 + i,
        name: `Player ${String(i + 1).padStart(4, "0")}`,
        n: i % 2 === 0 ? 5 : 2,
        sample: 1000 + i,
        pa: 400 + (i % 97),
        hr: 10 + (i % 23),
        ops: 1.1 - i * 0.0005,
        prev: i % 10 === 9 ? null : { pa: 500, hr: 20, ops: 0.75 },
      }),
    ),
  );
}

// ---------------------------------------------------------------------------
// 가짜 fetch · 화면 너비
// ---------------------------------------------------------------------------

const BAT_URL = `${DATA_BASE}/batters.json`;
const PIT_URL = `${DATA_BASE}/pitchers.json`;

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
// 화면 읽기 도우미
// ---------------------------------------------------------------------------

const norm = (s: string | null | undefined): string => (s ?? "").replace(/\s+/g, " ").trim();

function renderHome() {
  return render(
    <Providers>
      <HomePage />
    </Providers>,
  );
}

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

/** 역할 탭 — Astryx TabList(버튼+aria-current)·tab·radio 무엇이든 */
function roleTab(label: "타자" | "투수"): HTMLElement {
  const name = new RegExp(`^${label}`);
  const found = [
    ...screen.queryAllByRole("tab", { name }),
    ...screen.queryAllByRole("radio", { name }),
    ...screen.queryAllByRole("button", { name }),
  ];
  expect(found.length, `역할 탭 "${label}"`).toBeGreaterThan(0);
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

function prevToggle(): HTMLElement {
  const name = /2026 출전/;
  const found = [...screen.queryAllByRole("switch", { name }), ...screen.queryAllByRole("checkbox", { name })];
  expect(found.length, '"2026 출전 선수만" 토글').toBeGreaterThan(0);
  return found[0]!;
}

const isOn = (el: HTMLElement): boolean =>
  el.getAttribute("aria-checked") === "true" || (el instanceof HTMLInputElement && el.checked);

function searchBox(): HTMLInputElement {
  const name = /선수 이름 검색/;
  const found = [...screen.queryAllByRole("searchbox", { name }), ...screen.queryAllByRole("textbox", { name })];
  expect(found.length, "선수 이름 검색 입력").toBeGreaterThan(0);
  return found[0] as HTMLInputElement;
}

const search = (text: string) => fireEvent.change(searchBox(), { target: { value: text } });

/** 사용 시즌 필터 — Selector(콤보박스)·네이티브 select·SegmentedControl(radio) 무엇이든 */
async function chooseCareer(label: "전체" | "1~2시즌" | "3~5시즌") {
  const name = /사용 시즌/;
  const option = new RegExp(`${label}$`);
  const group = screen.queryAllByRole("radiogroup", { name })[0];
  if (group !== undefined) {
    fireEvent.click(within(group).getByRole("radio", { name: option }));
    return;
  }
  const combos = screen.queryAllByRole("combobox", { name });
  expect(combos.length, "사용 시즌 필터").toBeGreaterThan(0);
  const combo = combos[0]!;
  if (combo instanceof HTMLSelectElement) {
    const opt = Array.from(combo.options).find((o) => option.test(norm(o.textContent)));
    expect(opt, `사용 시즌 선택지 ${label}`).toBeDefined();
    fireEvent.change(combo, { target: { value: opt!.value } });
    return;
  }
  fireEvent.click(combo);
  fireEvent.click(await screen.findByRole("option", { name: option }));
}

/** 머리의 정렬 단추를 누른다 */
function clickSort(table: HTMLElement, re: RegExp) {
  fireEvent.click(within(headerOf(table, re)).getByRole("button"));
}

/** 머리의 aria-sort 방향대로 값이 놓였는지 */
function expectSortedBy(table: HTMLElement, re: RegExp, value: (id: number) => number) {
  const dir = headerOf(table, re).getAttribute("aria-sort");
  expect(["ascending", "descending"], `${re}의 aria-sort`).toContain(dir);
  const values = rowIds(table).map(value);
  const sorted = [...values].sort((a, b) => (dir === "ascending" ? a - b : b - a));
  expect(values).toEqual(sorted);
  return dir;
}

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

function loadingShown(): boolean {
  return screen.queryAllByText(/불러오는 중/).length > 0 || document.querySelector('[aria-busy="true"]') !== null;
}

/** 오류 안내가 보이는가 — Astryx 공용 알림 영역(role=alert)은 늘 있으므로 role만으로는 세지 않고 글자를 본다 */
function errorShown(): boolean {
  const re = /불러오지 못|받지 못|오류/;
  return (
    screen.queryAllByRole("alert").some((el) => re.test(norm(el.textContent))) || screen.queryAllByText(re).length > 0
  );
}

// ---------------------------------------------------------------------------
// 테스트
// ---------------------------------------------------------------------------

describe("예측표 — 타자 기본 보기 (F13, SA-1, DS2·DS3)", () => {
  it("타자 탭, '2026 출전 선수만' 켜짐, 2027 OPS 높은 순으로 시작한다", async () => {
    stubData({ [BAT_URL]: BAT, [PIT_URL]: PIT });
    renderHome();
    const table = await findTable();
    expect(isSelected(roleTab("타자"))).toBe(true);
    expect(isSelected(roleTab("투수"))).toBe(false);
    expect(isOn(prevToggle())).toBe(true);
    expect(rowIds(table)).toEqual(BAT_DEFAULT);
    expect(headerOf(table, /^2027 OPS/).getAttribute("aria-sort")).toBe("descending");
  });

  it("열: 선수 · 나이 · 사용 기록 · 2027 OPS · 리그 평균 대비 · PA* · HR* · 2026 OPS", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    const table = await findTable();
    const expected = [/^선수/, /^나이/, /^사용 기록/, /^2027 OPS/, /리그 평균/, /^PA\*/, /^HR\*/, /^2026 OPS/];
    const texts = headers(table).map((h) => norm(h.textContent));
    expect(texts).toHaveLength(expected.length);
    expected.forEach((re, i) => expect(texts[i]).toMatch(re));
    // 머리에 리그 평균 값을 적는다 — "리그 평균(.719) 대비"
    expect(headerOf(table, /리그 평균/)).toHaveTextContent(fmtRate(LEAGUE_OPS));
  });

  it("한 행: 사진·이름(상세 링크)·포지션·팀, 나이, 사용 기록, OPS ± 범위, 리그 평균 대비, PA·HR 정수, 2026 OPS", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    const table = await findTable();
    const s = batSpec(102);
    const player = cellOf(table, rowOf(table, 102), /^선수/);
    const links = within(player).getAllByRole("link");
    expect(links.some((a) => norm(a.textContent).includes("Yordan Alvarez"))).toBe(true);
    for (const a of links) expectHref(a, playerHref(102));
    expect(norm(player.textContent)).toMatch(/DH\s*·\s*Houston Astros/);
    expect(player).not.toHaveTextContent("현역 아님");

    expect(["30", "30세"]).toContain(cellText(table, 102, /^나이/));
    expect(cellText(table, 102, /^사용 기록/)).toMatch(/5시즌\s*·\s*2,583\s*타석/);

    const ops = cellText(table, 102, /^2027 OPS/);
    expect(ops).toContain(fmtRate(s.ops)); // .939
    expect(ops).not.toContain("0.939");
    expect(ops).toContain(fmtBand("ops", 0.096)); // ±.096

    expect(cellText(table, 102, /리그 평균/)).toContain(fmtDiff("ops", s.ops - LEAGUE_OPS)); // +.220
    expect(cellText(table, 102, /^PA\*/)).toBe(fmtCount(s.pa)); // 504
    expect(cellText(table, 102, /^HR\*/)).toBe(fmtCount(s.hr)); // 27
    expect(cellText(table, 102, /^2026 OPS/)).toBe(fmtRate(1.033));
  });

  it("± 범위는 경력 그룹 값이다 — 1~2시즌 ±.104, 3~5시즌 ±.096 (D29), 평균보다 낮으면 −차이", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    const table = await findTable();
    expect(cellText(table, 105, /^2027 OPS/)).toContain(fmtBand("ops", 0.104));
    expect(cellText(table, 105, /^2027 OPS/)).not.toContain(fmtBand("ops", 0.096));
    expect(cellText(table, 103, /^2027 OPS/)).toContain(fmtBand("ops", 0.096)); // 3시즌
    expect(cellText(table, 105, /리그 평균/)).toContain(fmtDiff("ops", batSpec(105).ops - LEAGUE_OPS)); // −.017
  });

  it("2026 OPS가 비어 있으면 —", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    const table = await findTable();
    expect(cellText(table, 108, /^2026 OPS/)).toBe(DASH);
  });

  it("표 아래에 ± 범위 설명과 출전량 설명을 단다 (D29·D30)", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    const table = await findTable();
    for (const note of [BAND_NOTE, PLAYING_TIME_NOTE]) {
      const found = screen.getAllByText((_, el) => {
        if (el === null || !norm(el.textContent).includes(note)) return false;
        return Array.from(el.children).every((c) => !norm(c.textContent).includes(note));
      });
      expect(found.length, note).toBeGreaterThan(0);
      expect(found.some((el) => table.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
    }
  });
});

describe("검색·필터 (F13, DS3)", () => {
  it("이름으로 찾는다 — 대소문자를 가리지 않고, 없으면 안내", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    await findTable();
    search("alvarez");
    await waitFor(() => expect(rowIds(getTable())).toEqual([102]));
    search("Witt");
    await waitFor(() => expect(rowIds(getTable())).toEqual([104]));
    search("zzzz");
    await waitFor(() => expect(screen.queryByRole("table") === null || rowIds(getTable()).length === 0).toBe(true));
    expect(screen.getAllByText(/없습니다/).length).toBeGreaterThan(0);
    search("");
    await waitFor(() => expect(rowIds(getTable())).toEqual(BAT_DEFAULT));
  });

  it("'2026 출전 선수만'을 끄면 2026에 뛰지 않은 선수도 나온다 (prev = null)", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    await findTable();
    fireEvent.click(prevToggle());
    await waitFor(() => expect(rowIds(getTable())).toEqual(BAT_ALL));
    expect(isOn(prevToggle())).toBe(false);
    expect(cellText(getTable(), 106, /^2026 OPS/)).toBe(DASH);
    fireEvent.click(prevToggle());
    await waitFor(() => expect(rowIds(getTable())).toEqual(BAT_DEFAULT));
  });

  it("'현역 아님'은 active=false인 선수에게만 붙는다 (D23)", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    await findTable();
    fireEvent.click(prevToggle());
    await waitFor(() => expect(rowIds(getTable())).toEqual(BAT_ALL));
    const table = getTable();
    expect(rowOf(table, 106)).toHaveTextContent("현역 아님");
    for (const id of BAT_ALL.filter((x) => x !== 106)) {
      expect(rowOf(table, id)).not.toHaveTextContent("현역 아님");
    }
  });

  it("팀·나이·포지션이 비어도 null·undefined·NaN을 쓰지 않는다", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    await findTable();
    fireEvent.click(prevToggle());
    await waitFor(() => expect(rowIds(getTable())).toContain(107));
    expect(norm(rowOf(getTable(), 107).textContent)).not.toMatch(/null|undefined|NaN/);
    expect(cellText(getTable(), 107, /^나이/)).toBe(DASH);
  });

  it("사용 시즌 필터 — 1~2시즌 / 3~5시즌 / 전체 (careerGroup)", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    await findTable();
    await chooseCareer("1~2시즌");
    await waitFor(() => expect(rowIds(getTable())).toEqual([105]));
    await chooseCareer("3~5시즌");
    await waitFor(() => expect(rowIds(getTable())).toEqual([102, 103, 101, 104, 108]));
    await chooseCareer("전체");
    await waitFor(() => expect(rowIds(getTable())).toEqual(BAT_DEFAULT));
    // 출전 필터와 함께 걸린다
    fireEvent.click(prevToggle());
    await chooseCareer("1~2시즌");
    await waitFor(() => expect(rowIds(getTable())).toEqual([105, 107]));
  });
});

describe("정렬 (F13)", () => {
  it("PA*·HR* 머리를 누르면 그 열로 정렬하고 aria-sort로 알린다", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    let table = await findTable();
    clickSort(table, /^HR\*/);
    await waitFor(() => expect(headerOf(getTable(), /^HR\*/).getAttribute("aria-sort")).toMatch(/ascending|descending/));
    table = getTable();
    expectSortedBy(table, /^HR\*/, (id) => batSpec(id).hr);
    expect(headerOf(table, /^2027 OPS/).getAttribute("aria-sort") ?? "none").toBe("none");
    expect(new Set(rowIds(table))).toEqual(new Set(BAT_DEFAULT)); // 정렬은 필터를 바꾸지 않는다

    clickSort(table, /^PA\*/);
    await waitFor(() => expect(headerOf(getTable(), /^PA\*/).getAttribute("aria-sort")).toMatch(/ascending|descending/));
    expectSortedBy(getTable(), /^PA\*/, (id) => batSpec(id).pa);
  });
});

describe("투수 탭 (F13)", () => {
  it("투수 표는 처음 누를 때 한 번만 받고, 2027 ERA 낮은 순으로 보인다", async () => {
    const fetchMock = stubData({ [BAT_URL]: BAT, [PIT_URL]: PIT });
    renderHome();
    await findTable();
    expect(callsTo(fetchMock, "/data/pitchers.json")).toBe(0);

    fireEvent.click(roleTab("투수"));
    await waitFor(() => expect(rowIds(getTable())).toEqual(PIT_DEFAULT), { timeout: 3000 });
    expect(isSelected(roleTab("투수"))).toBe(true);
    expect(headerOf(getTable(), /^2027 ERA/).getAttribute("aria-sort")).toBe("ascending");
    expect(callsTo(fetchMock, "/data/pitchers.json")).toBe(1);

    fireEvent.click(roleTab("타자"));
    await waitFor(() => expect(rowIds(getTable())).toEqual(BAT_DEFAULT));
    fireEvent.click(roleTab("투수"));
    await waitFor(() => expect(rowIds(getTable())).toEqual(PIT_DEFAULT));
    expect(callsTo(fetchMock, "/data/pitchers.json")).toBe(1);
    expect(callsTo(fetchMock, "/data/batters.json")).toBe(1);
  });

  it("열: 선수 · 나이 · 사용 기록(이닝) · 2027 ERA ± · 2027 K−BB% ± · IP* · 2026 ERA", async () => {
    stubData({ [BAT_URL]: BAT, [PIT_URL]: PIT });
    renderHome();
    await findTable();
    fireEvent.click(roleTab("투수"));
    await waitFor(() => expect(rowIds(getTable())).toEqual(PIT_DEFAULT), { timeout: 3000 });
    const table = getTable();
    const expected = [/^선수/, /^나이/, /^사용 기록/, /^2027 ERA/, /^2027 K[−-]BB%/, /^IP\*/, /^2026 ERA/];
    const texts = headers(table).map((h) => norm(h.textContent));
    expect(texts).toHaveLength(expected.length);
    expected.forEach((re, i) => expect(texts[i]).toMatch(re));

    const s = pitSpec(201);
    expect(cellText(table, 201, /^선수/)).toMatch(/P\s*·\s*Detroit Tigers/);
    expect(cellText(table, 201, /^사용 기록/)).toMatch(new RegExp(`5시즌\\s*·\\s*${fmtIp(s.sampleIp).replace(".", "\\.")}\\s*이닝`)); // 950.1이닝
    expect(cellText(table, 201, /^2027 ERA/)).toContain(fmtEra(s.era));
    expect(cellText(table, 201, /^2027 ERA/)).toContain(fmtBand("era", 1.38));
    expect(cellText(table, 201, /^2027 K[−-]BB%/)).toContain(fmtPct(s.kbb));
    expect(cellText(table, 201, /^2027 K[−-]BB%/)).toContain(fmtBand("k_bb_pct", 5.5));
    expect(cellText(table, 201, /^IP\*/)).toBe(fmtIp(s.ip)); // 172.1
    expect(cellText(table, 201, /^2026 ERA/)).toBe(fmtEra(2.45));

    // 1~2시즌 그룹의 범위
    expect(cellText(table, 203, /^사용 기록/)).toMatch(/2시즌\s*·\s*95\.2\s*이닝/);
    expect(cellText(table, 203, /^2027 ERA/)).toContain(fmtBand("era", 1.49));
    expect(cellText(table, 203, /^2027 K[−-]BB%/)).toContain(fmtBand("k_bb_pct", 5.9));
    expect(cellText(table, 203, /^IP\*/)).toBe(fmtIp(41.3)); // 41.1
    for (const a of within(rowOf(table, 201)).getAllByRole("link")) expectHref(a, playerHref(201));
  });

  it("투수도 K−BB% 머리로 정렬하고, 출전 필터를 끄면 미출전·현역 아님 투수가 나온다", async () => {
    stubData({ [BAT_URL]: BAT, [PIT_URL]: PIT });
    renderHome();
    await findTable();
    fireEvent.click(roleTab("투수"));
    await waitFor(() => expect(rowIds(getTable())).toEqual(PIT_DEFAULT), { timeout: 3000 });
    clickSort(getTable(), /^2027 K[−-]BB%/);
    await waitFor(() =>
      expect(headerOf(getTable(), /^2027 K[−-]BB%/).getAttribute("aria-sort")).toMatch(/ascending|descending/),
    );
    expectSortedBy(getTable(), /^2027 K[−-]BB%/, (id) => pitSpec(id).kbb);

    fireEvent.click(prevToggle());
    await waitFor(() => expect(rowIds(getTable())).toContain(205));
    expect(rowOf(getTable(), 205)).toHaveTextContent("현역 아님");
    expect(cellText(getTable(), 205, /^2026 ERA/)).toBe(DASH);
    expect(norm(rowOf(getTable(), 205).textContent)).not.toMatch(/null|undefined|NaN/);
  });

  it("투수 표를 받지 못하면 투수 탭에 오류를 보이고, 타자 탭은 그대로다", async () => {
    stubData({ [BAT_URL]: BAT }); // pitchers.json → 404
    renderHome();
    await findTable();
    fireEvent.click(roleTab("투수"));
    await waitFor(() => expect(errorShown()).toBe(true));
    fireEvent.click(roleTab("타자"));
    await waitFor(() => expect(rowIds(getTable())).toEqual(BAT_DEFAULT));
  });
});

describe(`페이지 나눔 (DS4 — 한 쪽 ${PAGE_SIZE}행)`, () => {
  const BIG = bigBatters(1144);
  const order = (rows: BatRow[]) =>
    [...rows].sort((a, b) => b.projection.ops - a.projection.ops).map((r) => r.player.player_id);
  const defaultOrder = order(BIG.rows.filter((r) => r.prev !== null));

  it("50행씩 보이고 2쪽으로 넘어간다", async () => {
    stubData({ [BAT_URL]: BIG });
    renderHome();
    const table = await findTable();
    expect(rowIds(table)).toEqual(defaultOrder.slice(0, PAGE_SIZE));
    goToPage(2);
    await waitFor(() => expect(rowIds(getTable())).toEqual(defaultOrder.slice(PAGE_SIZE, PAGE_SIZE * 2)));
  }, 20000);

  it("검색·출전 필터·사용 시즌을 바꾸면 1쪽으로 돌아간다", async () => {
    stubData({ [BAT_URL]: BIG });
    renderHome();
    await findTable();
    const page2 = defaultOrder.slice(PAGE_SIZE, PAGE_SIZE * 2);

    goToPage(2);
    await waitFor(() => expect(rowIds(getTable())).toEqual(page2));
    search("Player"); // 모두 맞는다 — 결과는 같고 쪽만 1로
    await waitFor(() => expect(rowIds(getTable())).toEqual(defaultOrder.slice(0, PAGE_SIZE)));

    goToPage(2);
    await waitFor(() => expect(rowIds(getTable())).toEqual(page2));
    fireEvent.click(prevToggle());
    await waitFor(() => expect(rowIds(getTable())).toEqual(order(BIG.rows).slice(0, PAGE_SIZE)));
    fireEvent.click(prevToggle());
    await waitFor(() => expect(rowIds(getTable())).toEqual(defaultOrder.slice(0, PAGE_SIZE)));

    goToPage(2);
    await waitFor(() => expect(rowIds(getTable())).toEqual(page2));
    await chooseCareer("3~5시즌");
    const veterans = order(BIG.rows.filter((r) => r.prev !== null && r.window.n_seasons >= 3));
    await waitFor(() => expect(rowIds(getTable())).toEqual(veterans.slice(0, PAGE_SIZE)));
  }, 30000);
});

describe("선수 사진 (D27)", () => {
  it("MLB 서버 사진을 쓰고, 못 불러오면 이니셜(initials)로 바꾼다", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    const table = await findTable();
    const cell = cellOf(table, rowOf(table, 104), /^선수/);
    const photos = mlbPhotos(cell);
    expect(photos).toHaveLength(1);
    // 주소는 playerPhotoUrl — 폭(w_)만 화면에 맞게 달라도 된다
    const anyWidth = (url: string) => url.replace(/w_\d+/, "w_N");
    expect(anyWidth(photos[0]!.getAttribute("src")!)).toBe(anyWidth(playerPhotoUrl(104)));

    fireEvent.error(photos[0]!);
    await waitFor(() => expect(mlbPhotos(cellOf(getTable(), rowOf(getTable(), 104), /^선수/))).toHaveLength(0));
    expect(showsInitials(cellOf(getTable(), rowOf(getTable(), 104), /^선수/), initials("Bobby Witt Jr."))).toBe(true); // "BW"
  });

  it("사진을 끄면(NEXT_PUBLIC_PLAYER_PHOTOS=off) 처음부터 이니셜만 보인다", async () => {
    vi.stubEnv("NEXT_PUBLIC_PLAYER_PHOTOS", "off");
    stubData({ [BAT_URL]: BAT });
    renderHome();
    const table = await findTable();
    expect(mlbPhotos(document.body)).toHaveLength(0);
    expect(showsInitials(cellOf(table, rowOf(table, 104), /^선수/), "BW")).toBe(true);
    expect(showsInitials(cellOf(table, rowOf(table, 102), /^선수/), "YA")).toBe(true);
  });
});

describe("모바일 (SA-3 — 390px)", () => {
  it("좁은 화면에서는 같은 순서의 선수 카드 목록 — 이름·포지션·팀·나이, 2027 OPS ±, PA*, HR*, 2026 OPS", async () => {
    setViewport(390);
    stubData({ [BAT_URL]: BAT });
    renderHome();
    const list = await waitFor(
      () => {
        const found = screen.getAllByRole("list").find((l) => playerLink(l) !== null);
        expect(found, "선수 카드 목록 (role=list)").toBeDefined();
        return found!;
      },
      { timeout: 3000 },
    );
    const cards = within(list)
      .getAllByRole("listitem")
      .filter((li) => playerLink(li) !== null);
    expect(cards.map(rowIdOf)).toEqual(BAT_DEFAULT);

    const card = cards[0]!;
    const s = batSpec(102);
    const text = norm(card.textContent);
    for (const a of within(card).getAllByRole("link")) expectHref(a, playerHref(102));
    expect(text).toContain("Yordan Alvarez");
    expect(text).toMatch(/DH\s*·\s*Houston Astros\s*·\s*30\s*세/);
    expect(text).toMatch(/2027 OPS/);
    expect(text).toContain(fmtRate(s.ops));
    expect(text).toContain(fmtBand("ops", 0.096));
    expect(text).toMatch(new RegExp(`PA\\*\\s*${fmtCount(s.pa)}`));
    expect(text).toMatch(new RegExp(`HR\\*\\s*${fmtCount(s.hr)}`));
    expect(text).toMatch(/2026 OPS\s*1\.033/);
  });
});

describe("불러오기 상태", () => {
  it("받는 동안 안내를 보이고, 다 받으면 표로 바꾼다", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    stubData({
      [BAT_URL]: async () => {
        await gate;
        return jsonResponse(BAT);
      },
    });
    renderHome();
    await waitFor(() => expect(loadingShown()).toBe(true));
    expect(screen.queryByRole("table")).toBeNull();
    release();
    const table = await findTable();
    expect(rowIds(table)).toEqual(BAT_DEFAULT);
    await waitFor(() => expect(screen.queryAllByText(/불러오는 중/)).toHaveLength(0));
  });

  it("받지 못하면(HTTP 500) 오류 안내를 보이고 표는 없다 — 머리글·바닥글은 그대로", async () => {
    stubData({ [BAT_URL]: async () => new Response("boom", { status: 500 }) });
    renderHome();
    await waitFor(() => expect(errorShown()).toBe(true), { timeout: 3000 });
    expect(screen.queryByRole("table")).toBeNull();
    expect(screen.getByRole("navigation", { name: "주요 메뉴" })).toBeInTheDocument();
    expect(document.querySelector("footer")).toHaveTextContent("MLB Stats API, Baseball Savant");
  });

  it("네트워크 오류(fetch 실패)도 오류 안내로 보인다", async () => {
    stubData({
      [BAT_URL]: async () => {
        throw new TypeError("network down");
      },
    });
    renderHome();
    await waitFor(() => expect(errorShown()).toBe(true), { timeout: 3000 });
    expect(screen.queryByRole("table")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// 레인 A Refactor — 빈 상태·다시 시도·첫 정렬 방향·모바일·주소 해시·D23, 순수 함수
// ---------------------------------------------------------------------------

/** 좁은 화면의 선수 카드 목록에서 선수 ID 순서 */
async function cardIds(): Promise<number[]> {
  return waitFor(
    () => {
      const list = screen.getAllByRole("list").find((l) => playerLink(l) !== null);
      expect(list, "선수 카드 목록").toBeDefined();
      return within(list!)
        .getAllByRole("listitem")
        .filter((li) => playerLink(li) !== null)
        .map(rowIdOf);
    },
    { timeout: 3000 },
  );
}

async function chooseOption(combo: RegExp, option: RegExp) {
  const combos = screen.queryAllByRole("combobox", { name: combo });
  expect(combos.length, `${combo} 선택`).toBeGreaterThan(0);
  fireEvent.click(combos[0]!);
  fireEvent.click(await screen.findByRole("option", { name: option }));
}

/** 처음 들어오는 주소 — 선수 상세에서 뒤로 가기로 돌아온 경우를 흉내 낸다 */
const arriveAt = (url: string) => window.history.replaceState(null, "", url);

describe("빈 결과·다시 시도 (F13)", () => {
  it("출전 필터 때문에 비었으면 몇 명이 숨었는지 알리고, 단추로 필터를 끈다", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    await findTable();
    search("retired"); // 106은 2026에 뛰지 않았다
    const heading = await screen.findByRole("heading", { level: 2, name: /조건에 맞는 선수가 없습니다/ });
    expect(heading).toBeInTheDocument();
    expect(screen.getAllByText(/끄면 1명이 더 있습니다/).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: /2026에 뛰지 않은 선수도 보기/ }));
    await waitFor(() => expect(rowIds(getTable())).toEqual([106]));
    expect(isOn(prevToggle())).toBe(false);
  });

  it("검색어 때문에 비었으면 '필터 지우기'로 첫 보기로 돌아간다", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    await findTable();
    fireEvent.click(prevToggle());
    search("zzzz");
    fireEvent.click(await screen.findByRole("button", { name: /필터 지우기/ }));
    await waitFor(() => expect(rowIds(getTable())).toEqual(BAT_DEFAULT));
    expect(searchBox().value).toBe("");
    expect(isOn(prevToggle())).toBe(true);
  });

  it("받지 못했을 때 '다시 시도'를 누르면 다시 받아 표를 보인다", async () => {
    let calls = 0;
    const fetchMock = stubData({
      [BAT_URL]: async () => (calls++ === 0 ? new Response("boom", { status: 500 }) : jsonResponse(BAT)),
    });
    renderHome();
    fireEvent.click(await screen.findByRole("button", { name: /다시 시도/ }, { timeout: 3000 }));
    const table = await findTable();
    expect(rowIds(table)).toEqual(BAT_DEFAULT);
    expect(callsTo(fetchMock, "/data/batters.json")).toBe(2);
    // Astryx 공용 알림 영역(role=alert)은 늘 있으므로 오류 안내 글자가 사라졌는지 본다
    expect(screen.queryAllByText(/불러오지 못/)).toHaveLength(0);
  });
});

describe("정렬 — 처음 누르면 좋은 쪽이 위로", () => {
  it("HR*는 많은 순, 나이는 적은 순으로 시작한다", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    await findTable();
    clickSort(getTable(), /^HR\*/);
    await waitFor(() => expect(headerOf(getTable(), /^HR\*/).getAttribute("aria-sort")).toBe("descending"));
    expect(rowIds(getTable())).toEqual([101, 102, 103, 104, 105, 108]);
    clickSort(getTable(), /^HR\*/); // 같은 열을 다시 누르면 뒤집는다
    await waitFor(() => expect(headerOf(getTable(), /^HR\*/).getAttribute("aria-sort")).toBe("ascending"));
    clickSort(getTable(), /^나이/);
    await waitFor(() => expect(headerOf(getTable(), /^나이/).getAttribute("aria-sort")).toBe("ascending"));
    expect(rowIds(getTable())).toEqual([105, 103, 104, 102, 108, 101]);
  });
});

describe("마지막 MLB 시즌 (D23)", () => {
  it("현역 아님이거나 2026에 뛰지 않은 선수에게 마지막 MLB 시즌을 붙이고, 2026 출전 현역은 붙이지 않는다", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    await findTable();
    fireEvent.click(prevToggle());
    await waitFor(() => expect(rowIds(getTable())).toEqual(BAT_ALL));
    const table = getTable();
    expect(cellText(table, 106, /^선수/)).toMatch(/현역 아님/);
    expect(cellText(table, 106, /^선수/)).toMatch(/마지막 MLB 시즌 2024/);
    expect(cellText(table, 107, /^선수/)).toMatch(/마지막 MLB 시즌 2025/); // 현역이지만 2026 미출전
    expect(cellText(table, 107, /^선수/)).not.toMatch(/현역 아님/);
    for (const id of BAT_DEFAULT) expect(cellText(table, id, /^선수/)).not.toMatch(/마지막 MLB 시즌/);
  });
});

describe("모바일 (SA-3) — 더 보기·정렬 선택", () => {
  it("50명씩 보이고 '50명 더 보기'로 이어 붙인다", async () => {
    setViewport(390);
    const BIG = bigBatters(1144);
    stubData({ [BAT_URL]: BIG });
    renderHome();
    const order = BIG.rows
      .filter((r) => r.prev !== null)
      .sort((a, b) => b.projection.ops - a.projection.ops)
      .map((r) => r.player.player_id);
    expect(await cardIds()).toEqual(order.slice(0, PAGE_SIZE));
    expect(screen.queryAllByRole("navigation", { name: /페이지/ })).toHaveLength(0); // 쪽 단추 대신 더 보기
    fireEvent.click(screen.getByRole("button", { name: `${fmtCount(PAGE_SIZE)}명 더 보기` }));
    await waitFor(async () => expect(await cardIds()).toEqual(order.slice(0, PAGE_SIZE * 2)));
  }, 20000);

  it("정렬 선택 — 'HR 많은 순'을 고르면 카드 순서가 바뀐다", async () => {
    setViewport(390);
    stubData({ [BAT_URL]: BAT });
    renderHome();
    expect(await cardIds()).toEqual(BAT_DEFAULT);
    await chooseOption(/^정렬/, /HR 많은 순/);
    await waitFor(async () => expect(await cardIds()).toEqual([101, 102, 103, 104, 105, 108]));
  });
});

describe("뒤로 가기 — 보기 상태를 주소 해시에 담는다 (DESIGN §1-2)", () => {
  it("검색·필터를 바꾸면 해시에 적고, 첫 보기로 돌아오면 해시를 지운다", async () => {
    stubData({ [BAT_URL]: BAT });
    renderHome();
    await findTable();
    expect(window.location.hash).toBe("");
    search("alvarez");
    await waitFor(() => expect(new URLSearchParams(window.location.hash.slice(1)).get("q")).toBe("alvarez"));
    fireEvent.click(prevToggle());
    await waitFor(() => expect(new URLSearchParams(window.location.hash.slice(1)).get("all")).toBe("1"));
    search("");
    fireEvent.click(prevToggle());
    await waitFor(() => expect(window.location.hash).toBe(""));
    expect(window.location.pathname).toBe("/");
  });

  it("해시로 돌아오면 투수 탭·출전 필터·정렬을 되살린다", async () => {
    arriveAt("/#role=pit&all=1&psort=kbb-desc");
    const fetchMock = stubData({ [BAT_URL]: BAT, [PIT_URL]: PIT });
    renderHome();
    await waitFor(() => expect(rowIds(getTable())).toContain(205), { timeout: 3000 });
    expect(isSelected(roleTab("투수"))).toBe(true);
    expect(isOn(prevToggle())).toBe(false);
    expect(headerOf(getTable(), /^2027 K[−-]BB%/).getAttribute("aria-sort")).toBe("descending");
    expectSortedBy(getTable(), /^2027 K[−-]BB%/, (id) => pitSpec(id).kbb);
    expect(callsTo(fetchMock, "/data/pitchers.json")).toBe(1);

    // 타자 탭으로 갔다 와도 투수 표를 다시 받지 않는다
    fireEvent.click(roleTab("타자"));
    await waitFor(() => expect(rowIds(getTable())).toEqual(BAT_ALL));
    fireEvent.click(roleTab("투수"));
    await waitFor(() => expect(rowIds(getTable())).toContain(205));
    expect(callsTo(fetchMock, "/data/pitchers.json")).toBe(1);
  });

  it("해시의 쪽 번호로 돌아온다 — 넓은 화면은 그 쪽, 좁은 화면은 그 묶음까지", async () => {
    const BIG = bigBatters(1144);
    const order = BIG.rows
      .filter((r) => r.prev !== null)
      .sort((a, b) => b.projection.ops - a.projection.ops)
      .map((r) => r.player.player_id);
    arriveAt("/#page=2");
    stubData({ [BAT_URL]: BIG });
    renderHome();
    await waitFor(() => expect(rowIds(getTable())).toEqual(order.slice(PAGE_SIZE, PAGE_SIZE * 2)), { timeout: 3000 });
    cleanup();

    setViewport(390);
    arriveAt("/#page=2");
    renderHome();
    expect(await cardIds()).toEqual(order.slice(0, PAGE_SIZE * 2));
  }, 30000);

  it("'#projections' 앵커는 보기 상태가 아니다 — 첫 보기로 시작하고 앵커를 지우지 않는다", async () => {
    arriveAt("/#projections");
    stubData({ [BAT_URL]: BAT });
    renderHome();
    expect(rowIds(await findTable())).toEqual(BAT_DEFAULT);
    expect(window.location.hash).toBe("#projections");
  });
});

describe("rows.ts·viewState.ts 순수 함수", () => {
  const views = toBatViews(BAT);

  it("검색은 대소문자·악센트를 무시한다 (Acuña ↔ acuna)", () => {
    expect(normalizeName("  Ronald Acuña Jr. ")).toBe("ronald acuna jr.");
    const withAccent = toBatViews(batTable([batRow({ ...BAT_SPECS[0]!, id: 900, name: "Ronald Acuña Jr." })]));
    expect(filterRows(withAccent, { query: "ACUNA", prevOnly: false, career: "all" }).map((v) => v.id)).toEqual([900]);
  });

  it("정렬 — 빈 값은 방향과 상관없이 맨 뒤, 같은 값은 이름순", () => {
    for (const direction of ["ascending", "descending"] as const) {
      const ids = sortRows(views, [{ sortKey: "age", direction }]).map((v) => v.id);
      expect(ids[ids.length - 1]).toBe(107); // 나이 null
      const prev = sortRows(views, [{ sortKey: "prevOps", direction }]).map((v) => v.id);
      expect(new Set(prev.slice(-3))).toEqual(new Set([106, 107, 108])); // 2026 OPS 없음
      expect(prev.slice(-3)).toEqual([107, 106, 108]); // Bench Bat · Retired Slugger · Zero Ops
    }
  });

  it("새 열은 좋은 쪽 방향, 같은 열은 받은 방향 그대로", () => {
    expect(nextSort([{ sortKey: "ops", direction: "descending" }], [{ sortKey: "hr", direction: "ascending" }])).toEqual([
      { sortKey: "hr", direction: "descending" },
    ]);
    expect(nextSort([{ sortKey: "ops", direction: "descending" }], [{ sortKey: "age", direction: "descending" }])).toEqual([
      { sortKey: "age", direction: "ascending" },
    ]);
    expect(nextSort([{ sortKey: "hr", direction: "descending" }], [{ sortKey: "hr", direction: "ascending" }])).toEqual([
      { sortKey: "hr", direction: "ascending" },
    ]);
  });

  it("쪽 수 — 비어도 1쪽, 659명이면 14쪽", () => {
    expect(pageCount(0, PAGE_SIZE)).toBe(1);
    expect(pageCount(50, PAGE_SIZE)).toBe(1);
    expect(pageCount(659, PAGE_SIZE)).toBe(14);
  });

  it("해시 — 기본값은 빈 문자열, 바꾼 값은 왕복, 틀린 값은 그 칸만 기본값", () => {
    expect(encodeView(DEFAULT_VIEW)).toBe("");
    const view = {
      ...DEFAULT_VIEW,
      role: "pit" as const,
      query: "Acuña Jr.",
      prevOnly: false,
      career: "seasons_1_2" as const,
      page: 3,
      batSort: [{ sortKey: "hr" as const, direction: "descending" as const }],
      pitSort: [{ sortKey: "kbb" as const, direction: "descending" as const }],
    };
    expect(decodeView(`#${encodeView(view)}`)).toEqual(view);
    expect(decodeView("#projections")).toBeNull();
    expect(decodeView("")).toBeNull();
    expect(isViewHash("#q=x")).toBe(true);
    expect(decodeView("#role=xx&page=-2&career=9&sort=bogus-desc&psort=era-up")).toEqual(DEFAULT_VIEW);
  });
});
