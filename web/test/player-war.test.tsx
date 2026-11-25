// 선수 상세 WAR 구역 (F22, 시안 SA-7 — DESIGN §4-1 WAR 표시 규칙) 컴포넌트 테스트 — M9 레인 B Red
// 데이터는 fetch를 바꿔 끼워 합성 샘플(tests/fixtures/synthetic/web)과 테스트 안에서 손으로 만든 war.json으로 준다.
// 네트워크는 쓰지 않는다. 실존 선수 이름은 쓰지 않는다.
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PlayerPage from "@/app/player/page";
import { Providers } from "@/app/providers";
import { DATA_BASE } from "@/src/data/load";
import type { PitSeason, PlayerDetail, WarFile, WarRow } from "@/src/data/types";
import { fmtWar, fmtWarSplit } from "@/src/format";
import {
  BAT_RUNS_PARTS,
  FULL_TIME_ADJUSTED,
  FULL_TIME_NAME,
  PIT_WAR_FIP_COL,
  PIT_WAR_NOTE,
  PIT_WAR_RA9_COL,
  PIT_WAR_TITLE,
  RUNS_NOT_EXACT,
  WAR_LABEL,
  WAR_NOT_IN_FILE,
  runsPerWinNote,
} from "@/src/warText";

// ---------------------------------------------------------------------------
// 주소(?id=)와 브라우저 대역 — test/player.test.tsx와 같은 짜임
// ---------------------------------------------------------------------------

const nav = vi.hoisted(() => ({ params: new URLSearchParams() }));
vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useSearchParams: () => nav.params,
  usePathname: () => "/player/",
  useRouter: () => ({
    push: () => undefined,
    replace: () => undefined,
    back: () => undefined,
    forward: () => undefined,
    refresh: () => undefined,
    prefetch: () => undefined,
  }),
}));

function setSearch(search: string): void {
  nav.params = new URLSearchParams(search);
  window.history.replaceState(null, "", `/player/${search}`);
}

// Astryx Chart는 ResizeObserver로 폭을 재야 그린다. jsdom에는 없어서 폭 800을 알려 주는 대역을 둔다.
class FixedResizeObserver {
  private readonly callback: ResizeObserverCallback;

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
  }

  observe(target: Element): void {
    const entry = { target, contentRect: { width: 800, height: 300 } } as unknown as ResizeObserverEntry;
    this.callback([entry], this as unknown as ResizeObserver);
  }

  unobserve(): void {}

  disconnect(): void {}
}

type Reply = PlayerDetail | WarFile | { status: number } | "reject" | "pending";

const WAR_PATH = `${DATA_BASE}/war.json`;

function playerPath(id: number): string {
  return `${DATA_BASE}/players/${id}.json`;
}

/** 경로 → 응답. 모르는 경로는 404 */
function stubFetch(routes: Record<string, Reply>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
    const reply = routes[String(input)];
    if (reply === undefined) return new Response("not found", { status: 404 });
    if (reply === "reject") throw new TypeError("Failed to fetch");
    if (reply === "pending") return new Promise<Response>(() => undefined);
    if ("status" in reply && typeof reply.status === "number") return new Response("error", { status: reply.status });
    return new Response(JSON.stringify(reply), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function openPage(search: string, routes: Record<string, Reply>) {
  setSearch(search);
  const fetchMock = stubFetch(routes);
  const view = render(
    <Providers>
      <PlayerPage />
    </Providers>,
  );
  return { fetchMock, ...view };
}

function requested(fetchMock: ReturnType<typeof stubFetch>): string[] {
  return fetchMock.mock.calls.map(([url]) => String(url));
}

/** 선수 상세를 열고 이름(h1)이 뜰 때까지 기다린다 (war.json 응답은 war로 준다) */
async function openPlayer(detail: PlayerDetail, war: Reply) {
  const opened = openPage(`?id=${detail.player.player_id}`, {
    [playerPath(detail.player.player_id)]: detail,
    [WAR_PATH]: war,
  });
  await screen.findByRole("heading", { level: 1, name: (name) => name.startsWith(detail.player.name) });
  return opened;
}

/** WAR 구역(카드 또는 '2026 출전 선수만' 안내)이 뜰 때까지 기다린다 */
async function openWar(detail: PlayerDetail, war: WarFile = HAND_WAR) {
  const opened = await openPlayer(detail, war);
  await waitFor(() => expect(mainText()).toMatch(/2027 예측 WAR\s*\*|WAR 예측은 2026 출전 선수만/));
  return opened;
}

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", FixedResizeObserver);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  window.history.replaceState(null, "", "/");
});

// ---------------------------------------------------------------------------
// 화면 읽기 도우미
// ---------------------------------------------------------------------------

function headingLevel(el: Element): number | null {
  const match = /^H([1-6])$/.exec(el.tagName);
  if (match) return Number(match[1]);
  if (el.getAttribute("role") === "heading") return Number(el.getAttribute("aria-level") ?? "2");
  return null;
}

function clean(text: string | null | undefined): string {
  return (text ?? "").replace(/\s+/g, " ").trim();
}

function isBetween(node: Node, start: Node | null, end: Node | null): boolean {
  if (start !== null) {
    if (start.contains(node)) return false;
    if (!(start.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING)) return false;
  }
  if (end !== null && !(end.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_PRECEDING)) return false;
  return true;
}

function textBetween(root: Node, start: Node | null, end: Node | null): string {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let out = "";
  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) {
    if (isBetween(node, start, end)) out += node.textContent ?? "";
  }
  return clean(out);
}

function main(): HTMLElement {
  return screen.getByRole("main");
}

function mainText(): string {
  return clean(main().textContent);
}

/** 본문 안의 불러오는 중 표시 (Astryx 알림 영역은 본문 밖이라 세지 않는다) */
function warLoadingShown(): boolean {
  return within(main()).queryAllByRole("progressbar").length > 0 || /불러오는 중/.test(mainText());
}

interface Section {
  text: string;
  tables: HTMLTableElement[];
  charts: SVGSVGElement[];
}

/** 제목부터 같은 수준 이하의 다음 제목 전까지 */
function sectionOf(name: RegExp | string): Section {
  const heading = screen.getByRole("heading", { name });
  const level = headingLevel(heading) ?? 2;
  const root = heading.closest("main") ?? document.body;
  const stop =
    Array.from(root.querySelectorAll("*")).find((el) => {
      const l = headingLevel(el);
      return l !== null && l <= level && isBetween(el, heading, null);
    }) ?? null;
  const pick = <T extends Element>(selector: string): T[] =>
    Array.from(root.querySelectorAll<T>(selector)).filter((el) => isBetween(el, heading, stop));
  return { text: textBetween(root, heading, stop), tables: pick("table"), charts: pick('svg[role="img"]') };
}

interface Grid {
  headers: string[];
  rows: string[][];
}

function readTable(table: HTMLTableElement): Grid {
  const headRow = table.tHead?.rows[0] ?? table.rows[0];
  const headers = headRow ? Array.from(headRow.cells).map((c) => clean(c.textContent)) : [];
  const bodyRows = Array.from(table.tBodies).flatMap((body) => Array.from(body.rows));
  return { headers, rows: bodyRows.map((row) => Array.from(row.cells).map((c) => clean(c.textContent))) };
}

function col(grid: Grid, header: string | RegExp): number {
  const index = grid.headers.findIndex((h) => (typeof header === "string" ? h === header : header.test(h)));
  if (index < 0) throw new Error(`${String(header)} 열이 없습니다: ${grid.headers.join(" | ")}`);
  return index;
}

function cell(grid: Grid, season: number, header: string | RegExp): string {
  const seasonCol = col(grid, "시즌");
  const row = grid.rows.find((r) => (r[seasonCol] ?? "").startsWith(String(season)));
  if (row === undefined) throw new Error(`${season} 행이 없습니다`);
  return row[col(grid, header)] ?? "";
}

/** 정규식 특수 문자 이스케이프 ("+22.4" 같은 값을 그대로 찾는다) */
function esc(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** 차트 이미지의 설명 (aria-describedby — Astryx Chart는 subtitle을 <desc>로 잇는다). 조상의 설명도 모은다 */
function describedText(el: Element): string {
  const parts: string[] = [];
  for (let node: Element | null = el; node !== null; node = node.parentElement) {
    for (const id of (node.getAttribute("aria-describedby") ?? "").split(/\s+/).filter(Boolean)) {
      parts.push(document.getElementById(id)?.textContent ?? "");
    }
  }
  return clean(parts.join(" "));
}

/** 설명 글을 시즌별로 자른다 — 시즌이 오래된 순으로 나와야 한다 */
function seasonSlices(text: string, seasons: number[]): Map<number, string> {
  const starts: number[] = [];
  let from = 0;
  for (const season of seasons) {
    const at = text.indexOf(String(season), from);
    if (at < 0) throw new Error(`설명에 ${season}이(가) 차례대로 없습니다: ${text}`);
    starts.push(at);
    from = at + 4;
  }
  return new Map(seasons.map((season, i) => [season, text.slice(starts[i], starts[i + 1])]));
}

interface WarCards {
  /** ① 2027 예측 WAR* */
  pred: string;
  /** ② 풀타임 환산 WAR */
  full: string;
  /** ③ 예측 출전* */
  time: string;
}

/** WAR 카드 셋의 글자 — 「2027 예측 WAR*」 → 「풀타임 환산 WAR」 → 「예측 출전*」 → 「시즌별 WAR」 제목 차례 (SA-7) */
function warCards(): WarCards {
  const text = mainText();
  const at = (re: RegExp, from: number): number => {
    const i = text.slice(from).search(re);
    if (i < 0) throw new Error(`${String(re)}이(가) 없습니다: ${text.slice(from, from + 400)}`);
    return from + i;
  };
  const pred = at(/2027 예측 WAR\s*\*/, 0);
  const full = at(new RegExp(esc(FULL_TIME_NAME)), pred + 1);
  const time = at(/예측 출전\s*\*/, full + 1);
  const end = at(/시즌별 WAR/, time + 1);
  return { pred: text.slice(pred, full), full: text.slice(full, time), time: text.slice(time, end) };
}

function roleControl(name: string): HTMLElement | null {
  return (
    screen.queryByRole("radio", { name }) ??
    screen.queryByRole("tab", { name }) ??
    screen.queryByRole("button", { name })
  );
}

// ---------------------------------------------------------------------------
// 픽스처 — 합성 샘플 + 손으로 만든 PlayerDetail·WarFile (타입을 그대로 지킨다)
// ---------------------------------------------------------------------------

const ROOT = join(import.meta.dirname, "..", "..");

function sample<T>(name: string): T {
  return JSON.parse(readFileSync(join(ROOT, "tests/fixtures/synthetic/web", name), "utf-8")) as T;
}

/** 합성 타자(id 1): 2020·2026 두 시즌, 2026 655타석, 예측 610.4타석 */
const SAMPLE_BAT = sample<PlayerDetail>("player_bat.json");
/** 합성 투수(id 2): 2026 미출전 — 손으로 만든 war.json에 없다 (D42) */
const SAMPLE_PIT = sample<PlayerDetail>("player_pit.json");
/** 합성 war.json: id 1 투타 겸업(SA-7 숫자 그대로) · id 2 불펜 */
const SAMPLE_WAR = sample<WarFile>("war.json");

function pitSeason(season: number, over: Partial<PitSeason>): PitSeason {
  return { ...SAMPLE_PIT.pit!.history[0]!, season, team_name: "Sample Team", ...over };
}

/** 투타 겸업 (id 3): 타자 2026 655타석 · 예측 610.4타석, 투수 2026 150.667이닝 · 예측 160.3이닝 */
const TWO_WAY: PlayerDetail = {
  meta: SAMPLE_BAT.meta,
  player: {
    player_id: 3,
    name: "Two Way Sample",
    birth_date: "1994-07-05",
    primary_position: "TWP",
    bats: "L",
    throws: "R",
    active: true,
    last_mlb_season: 2026,
  },
  bat: SAMPLE_BAT.bat,
  pit: {
    window: { input_first: 2022, input_last: 2026, n_seasons: 4 },
    sample_bf: 2400,
    sample_ip: 580.3,
    projection: { ip: 160.3, era: 3.42, k_bb_pct: 19.6 },
    prev: { season: 2026, ip: 150.667, era: 3.1, k_bb_pct: 19.5 },
    league_baseline: { era: 4.15, k_bb_pct: 13.6 },
    group_error: {
      group: "seasons_3_5",
      mae: { era: 0.843, ip: 17.13, k_bb_pct: 3.62 },
      band80: { era: 1.38, k_bb_pct: 5.5 },
    },
    history: [
      pitSeason(2025, { g: 14, gs: 14, ip: 47.0, era: 2.87 }),
      pitSeason(2026, { g: 28, gs: 28, ip: 150.667, era: 3.1 }),
    ],
  },
  fielding: [],
};

/** 타자만 (id 1, 합성 타자 그대로) */
const BATTER: PlayerDetail = SAMPLE_BAT;

/** 불펜 투수만 (id 7): 2026 61.667이닝 · 예측 58.4이닝 */
const RELIEVER: PlayerDetail = {
  meta: SAMPLE_PIT.meta,
  player: { ...SAMPLE_PIT.player, player_id: 7, name: "Sample Reliever", active: true, last_mlb_season: 2026 },
  bat: null,
  pit: {
    ...SAMPLE_PIT.pit!,
    window: { input_first: 2025, input_last: 2026, n_seasons: 2 },
    projection: { ip: 58.4, era: 3.61, k_bb_pct: 16.2 },
    prev: { season: 2026, ip: 61.667, era: 3.2, k_bb_pct: 15.1 },
    history: [SAMPLE_PIT.pit!.history[0]!, pitSeason(2026, { g: 64, gs: 0, ip: 61.667, era: 3.2 })],
  },
  fielding: [],
};

/** 2026에 뛰지 않은 투수 (id 2) — war.json에 행이 없다 */
const ABSENT: PlayerDetail = SAMPLE_PIT;

function warRow(detail: PlayerDetail, over: Omit<Partial<WarRow>, "player" | "played_prev">): WarRow {
  return {
    player: {
      player_id: detail.player.player_id,
      name: detail.player.name,
      primary_position: detail.player.primary_position,
      age: 30,
      team_name: "Sample Team",
      active: detail.player.active,
      last_mlb_season: detail.player.last_mlb_season,
    },
    roles: "bat",
    played_prev: true,
    group: "seasons_3_5",
    slot: null,
    pa: null,
    ip: null,
    war_bat: null,
    war_pit: null,
    war: 0,
    war_full_bat: null,
    war_full_pit: null,
    war_full: 0,
    prev_war: null,
    prev_bat_runs: null,
    history: [],
    ...over,
  };
}

/**
 * 투타 겸업 WAR: 예측 5.98(타 4.12 + 투 1.86) · 풀타임 6.28(타 4.07 + 투 2.21, 선발) · 2026 실제 5.27(3.14 + 2.13)
 * 구성 항목 합 30.34점(수비 0.04 → "0.0") — 2023은 투수 WAR이 없다, 2024 실점 기반은 음수
 */
const W_TWO_WAY = warRow(TWO_WAY, {
  roles: "both",
  slot: "sp",
  pa: 610.4,
  ip: 160.3,
  war_bat: 4.12,
  war_pit: 1.86,
  war: 5.98,
  war_full_bat: 4.07,
  war_full_pit: 2.21,
  war_full: 6.28,
  prev_war: 5.27,
  prev_bat_runs: {
    runs_batting: 22.4,
    runs_baserunning: -1.6,
    runs_fielding: 0.04,
    runs_positional: -7.5,
    runs_league: 0.9,
    runs_replacement: 16.1,
  },
  history: [
    { season: 2022, war_bat: 1.24, war_pit: 3.47, war_pit_ra9: 2.91 },
    { season: 2023, war_bat: 4.06, war_pit: null, war_pit_ra9: null },
    { season: 2024, war_bat: 5.52, war_pit: 0.48, war_pit_ra9: -0.31 },
    { season: 2025, war_bat: 2.83, war_pit: 1.66, war_pit_ra9: 1.94 },
    { season: 2026, war_bat: 3.14, war_pit: 2.13, war_pit_ra9: 2.71 },
  ],
});

/** 타자 WAR: 예측 3.42 · 풀타임 3.36 · 2026 실제 4.18, 구성 항목 합 45.0점 */
const W_BATTER = warRow(BATTER, {
  roles: "bat",
  pa: 610.4,
  war_bat: 3.42,
  war: 3.42,
  war_full_bat: 3.36,
  war_full: 3.36,
  prev_war: 4.18,
  prev_bat_runs: {
    runs_batting: 18.3,
    runs_baserunning: 2.6,
    runs_fielding: -3.2,
    runs_positional: 7.5,
    runs_league: 0.4,
    runs_replacement: 19.4,
  },
  history: [
    { season: 2025, war_bat: 2.57, war_pit: null, war_pit_ra9: null },
    { season: 2026, war_bat: 4.18, war_pit: null, war_pit_ra9: null },
  ],
});

/** 불펜 WAR: 예측 0.83 · 풀타임(65이닝) 0.92 · 2026 실제 1.07, 타자 구성 항목 없음 */
const W_RELIEVER = warRow(RELIEVER, {
  roles: "pit",
  group: "seasons_1_2",
  slot: "rp",
  ip: 58.4,
  war_pit: 0.83,
  war: 0.83,
  war_full_pit: 0.92,
  war_full: 0.92,
  prev_war: 1.07,
  history: [
    { season: 2025, war_bat: null, war_pit: 0.36, war_pit_ra9: 0.52 },
    { season: 2026, war_bat: null, war_pit: 1.07, war_pit_ra9: -0.18 },
  ],
});

/** 손으로 만든 war.json — 합성 샘플의 meta·full_time·evaluations에 행 셋, 점 → 승 비율 10.37 */
const HAND_WAR: WarFile = {
  ...SAMPLE_WAR,
  runs_per_win: { season: 2026, value: 10.37 },
  rows: [W_TWO_WAY, W_BATTER, W_RELIEVER],
};

// ---------------------------------------------------------------------------
// 테스트
// ---------------------------------------------------------------------------

describe("war.json 받기 (N6 · D42)", () => {
  it("선수 상세를 먼저 받고, 그 뒤 war.json을 한 번 받는다", async () => {
    const { fetchMock } = await openWar(TWO_WAY);
    await waitFor(() => expect(requested(fetchMock)).toEqual([playerPath(3), WAR_PATH]));
  });

  it("선수 상세를 받기 전에는 war.json을 받지 않는다", async () => {
    const { fetchMock } = openPage("?id=3", { [playerPath(3)]: "pending", [WAR_PATH]: HAND_WAR });
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(requested(fetchMock)).toEqual([playerPath(3)]);
  });

  it("없는 선수(404)·서버 오류면 war.json을 받지 않는다", async () => {
    const missing = openPage("?id=999", { [WAR_PATH]: HAND_WAR });
    await waitFor(() => expect(mainText()).toMatch(/찾을 수 없/));
    expect(requested(missing.fetchMock)).toEqual([playerPath(999)]);
    cleanup();
    const broken = openPage("?id=3", { [playerPath(3)]: { status: 500 }, [WAR_PATH]: HAND_WAR });
    await waitFor(() => expect(mainText()).toMatch(/불러오지 못/));
    expect(requested(broken.fetchMock)).toEqual([playerPath(3)]);
  });

  it("war.json을 받는 동안 기존 상세는 그대로 보이고 WAR 자리에 불러오는 중 표시", async () => {
    await openPlayer(TWO_WAY, "pending");
    expect(mainText()).toMatch(/2027 OPS/);
    await waitFor(() => expect(warLoadingShown()).toBe(true));
    expect(mainText()).not.toMatch(/2027 예측 WAR/);
  });

  it("다 받으면 본문에 불러오는 중 표시가 남지 않는다", async () => {
    await openWar(TWO_WAY);
    expect(warLoadingShown()).toBe(false);
  });

  for (const [label, reply] of [
    ["서버 오류 500", { status: 500 }],
    ["네트워크 오류", "reject"],
  ] as const) {
    it(`war.json ${label}: 기존 상세는 그대로, WAR 자리에 '불러오지 못함' 안내 (2026 미출전 안내와 다르다)`, async () => {
      await openPlayer(TWO_WAY, reply);
      await waitFor(() => expect(mainText()).toMatch(/불러오지 못/));
      expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Two Way Sample");
      expect(mainText()).toMatch(/2027 OPS/);
      expect(mainText()).not.toMatch(/2027 예측 WAR/);
      expect(mainText()).not.toContain(WAR_NOT_IN_FILE);
      expect(mainText()).not.toMatch(/NaN|undefined|null|Infinity/);
    });
  }

  it("2026에 뛰지 않은 선수(war.json에 없음)는 WAR 구역 대신 안내 한 줄 (D42)", async () => {
    const { fetchMock } = await openWar(ABSENT);
    expect(requested(fetchMock)).toContain(WAR_PATH);
    expect(screen.getAllByText(WAR_NOT_IN_FILE)).toHaveLength(1);
    const text = mainText();
    expect(text).not.toMatch(/2027 예측 WAR/);
    expect(text).not.toContain(FULL_TIME_NAME);
    expect(text).not.toMatch(/예측 출전/);
    expect(screen.queryByRole("heading", { name: /시즌별 WAR/ })).toBeNull();
    expect(screen.queryByRole("heading", { name: /이렇게 나왔다/ })).toBeNull();
    expect(screen.queryByRole("heading", { name: PIT_WAR_TITLE })).toBeNull();
    // 기존 상세는 그대로
    expect(text).toMatch(/2027 ERA\s*4\.05/);
    expect(screen.getByRole("heading", { name: /최근 5시즌/ })).toBeInTheDocument();
  });

  it("투타 겸업에서 역할을 바꿔도 WAR 구역은 그대로이고 war.json을 다시 받지 않는다", async () => {
    const { fetchMock } = await openWar(TWO_WAY);
    const before = warCards();
    fireEvent.click(roleControl("투수")!);
    await waitFor(() => expect(mainText()).toMatch(/2027 ERA/));
    expect(warCards()).toEqual(before);
    expect(requested(fetchMock).filter((url) => url === WAR_PATH)).toHaveLength(1);
  });
});

describe("WAR 카드 셋 (SA-7, D30 · D34 · D35 · D41)", () => {
  it("WAR 카드는 기존 2027 예측 카드 아래에 있다", async () => {
    await openWar(TWO_WAY);
    const text = mainText();
    expect(text.indexOf("2027 OPS")).toBeGreaterThanOrEqual(0);
    expect(text.search(/2027 예측 WAR\s*\*/)).toBeGreaterThan(text.indexOf("2027 OPS"));
  });

  it("투타 겸업 ① 예측 WAR*: 합(fmtWar)을 크게, 아래 '타 + 투', 2026 실제 합(타 + 투) · 보수적 예상", async () => {
    await openWar(TWO_WAY);
    const { pred } = warCards();
    expect(pred).toMatch(/^2027 예측 WAR\s*\*\s*6\.0/); // fmtWar(5.98)
    expect(pred).toContain("타 4.1 + 투 1.9"); // fmtWarSplit(4.12, 1.86)
    expect(pred).toMatch(/2026 실제\s*5\.3\s*\(\s*3\.1\s*\+\s*2\.1\s*\)\s*·\s*보수적 예상/); // prev_war · 2026 war_bat · war_pit
  });

  it("투타 겸업 ② 풀타임 환산 WAR: 합 · 투타 분해 · 타자 600타석 · 선발 180이닝 기준 · 조정 WAR 설명 · 드물다", async () => {
    await openWar(TWO_WAY);
    const { full } = warCards();
    expect(full).toMatch(new RegExp(`^${esc(FULL_TIME_NAME)}\\s*6\\.3`)); // fmtWar(6.28)
    expect(full).toMatch(/4\.1[\s\S]*2\.2/); // war_full_bat 4.07 → 4.1, war_full_pit 2.21 → 2.2 (타자 먼저)
    expect(full).toMatch(/타자 600타석/);
    expect(full).toMatch(/선발 180이닝/);
    expect(full).not.toMatch(/불펜/); // 그 선수 역할의 기준만
    expect(full).toContain(FULL_TIME_ADJUSTED);
    expect(full).toContain("두 역할 모두 풀타임인 경우는 드물다");
  });

  it("투타 겸업 ③ 예측 출전*: 타석 · 이닝(fmtCount 반올림), 2026 실제 타석 · 이닝(선수 상세 기록)", async () => {
    await openWar(TWO_WAY);
    const { time } = warCards();
    expect(time).toMatch(/^예측 출전\s*\*\s*610\s*타석\s*투수\s*160\s*이닝\s*·\s*선발로 분류/); // 큰 숫자는 타석만, 투수 몫은 아래 줄 (SA-7) — fmtCount(160.3), 야구 이닝 표기(160.1) 아님
    expect(time).toMatch(/2026 실제\s*655\s*타석\s*·\s*151\s*이닝/); // bat history 2026 PA · pit history 2026 IP 150.667
  });

  it("타자만: 예측 WAR에 투타 분해가 없고 2026 실제에 괄호가 없다 · 기준은 타자 600타석만", async () => {
    await openWar(BATTER);
    const { pred, full, time } = warCards();
    expect(pred).toMatch(/^2027 예측 WAR\s*\*\s*3\.4/); // fmtWar(3.42)
    expect(pred).not.toMatch(/타 \S+ \+ 투/);
    expect(pred).toMatch(/2026 실제\s*4\.2\s*·\s*보수적 예상/); // fmtWar(4.18)
    expect(pred).not.toContain("(");
    expect(full).toMatch(new RegExp(`^${esc(FULL_TIME_NAME)}\\s*3\\.4`)); // fmtWar(3.36)
    expect(full).toMatch(/타자 600타석/);
    expect(full).not.toMatch(/선발|불펜/);
    expect(full).not.toContain("두 역할 모두 풀타임");
    expect(full).toContain(FULL_TIME_ADJUSTED);
    expect(time).toMatch(/^예측 출전\s*\*\s*610\s*타석/);
    expect(time).not.toMatch(/이닝/);
    expect(time).toMatch(/2026 실제\s*655\s*타석/);
  });

  it("불펜 투수: 기준은 불펜 65이닝만, 예측 출전은 이닝만", async () => {
    await openWar(RELIEVER);
    const { pred, full, time } = warCards();
    expect(pred).toMatch(/^2027 예측 WAR\s*\*\s*0\.8/); // fmtWar(0.83)
    expect(pred).toMatch(/2026 실제\s*1\.1\s*·\s*보수적 예상/); // fmtWar(1.07)
    expect(pred).not.toContain("(");
    expect(full).toMatch(new RegExp(`^${esc(FULL_TIME_NAME)}\\s*0\\.9`)); // fmtWar(0.92)
    expect(full).toMatch(/불펜 65이닝/);
    expect(full).not.toMatch(/선발|타자 600타석/);
    expect(full).not.toContain("두 역할 모두 풀타임");
    expect(time).toMatch(/^예측 출전\s*\*\s*58\s*이닝\s*불펜으로 분류/); // fmtCount(58.4)
    expect(time).not.toMatch(/타석/);
    expect(time).toMatch(/2026 실제\s*62\s*이닝/); // fmtCount(61.667)
  });

  it("합성 war.json 그대로 (SA-7 숫자): 예측 · 풀타임 · 출전", async () => {
    const twoWay1: PlayerDetail = { ...TWO_WAY, player: { ...TWO_WAY.player, player_id: 1, name: "Sample Batter" } };
    await openWar(twoWay1, SAMPLE_WAR);
    const { pred, full, time } = warCards();
    expect(pred).toMatch(new RegExp(`^2027 예측 WAR\\s*\\*\\s*${esc(fmtWar(5.97))}`));
    expect(pred).toContain(fmtWarSplit(4.12, 1.85));
    expect(pred).toMatch(/2026 실제\s*6\.6\s*\(\s*3\.7\s*\+\s*2\.9\s*\)\s*·\s*보수적 예상/);
    expect(full).toMatch(new RegExp(`^${esc(FULL_TIME_NAME)}\\s*9\\.3`)); // fmtWar(9.29)
    expect(full).toMatch(new RegExp(`${esc(fmtWar(4.95))}[\\s\\S]*${esc(fmtWar(4.34))}`));
    expect(time).toMatch(/^예측 출전\s*\*\s*500\s*타석\s*투수\s*77\s*이닝\s*·\s*선발로 분류/); // fmtCount(499.8) · fmtCount(76.7)
  });
});

describe("시즌별 WAR — 타 · 투 쌓은 막대 (SA-7)", () => {
  it("투타 겸업: 차트(svg) 설명에 시즌마다 타자 · 투수 WAR(fmtWar)을 오래된 순으로 읽어 준다", async () => {
    await openWar(TWO_WAY);
    const section = sectionOf(/시즌별 WAR/);
    const chart = section.charts[0];
    expect(chart, `시즌별 WAR 구역에 차트가 없습니다: ${section.text.slice(0, 200)}`).toBeDefined();
    expect(chart!.tagName.toLowerCase()).toBe("svg");
    expect(chart!.getAttribute("aria-label") ?? "").toMatch(/WAR/);
    const desc = describedText(chart!);
    const slices = seasonSlices(desc, [2022, 2023, 2024, 2025, 2026]);
    const expected: [number, string[]][] = [
      [2022, ["1.2", "3.5"]],
      [2023, ["4.1"]], // 투수 WAR 없음
      [2024, ["5.5", "0.5"]],
      [2025, ["2.8", "1.7"]],
      [2026, ["3.1", "2.1"]],
    ];
    for (const [season, values] of expected) {
      for (const v of values) expect(slices.get(season), `${season} 설명에 ${v}이(가) 없습니다: ${desc}`).toContain(v);
    }
    expect(desc).not.toMatch(/NaN|undefined|null/);
  });

  it("타자만: 시즌별 타자 WAR 차트", async () => {
    await openWar(BATTER);
    const chart = sectionOf(/시즌별 WAR/).charts[0];
    expect(chart).toBeDefined();
    const slices = seasonSlices(describedText(chart!), [2025, 2026]);
    expect(slices.get(2025)).toContain("2.6"); // fmtWar(2.57)
    expect(slices.get(2026)).toContain("4.2"); // fmtWar(4.18)
  });

  /** 차트 왼쪽 축 눈금 값 (fmtWar 글자 → 숫자) */
  function tickValues(chart: SVGSVGElement): number[] {
    return Array.from(chart.querySelectorAll("text"))
      .map((t) => clean(t.textContent))
      .filter((t) => /^[−-]?\d+\.\d$/.test(t))
      .map((t) => Number(t.replace("−", "-")));
  }

  it("풀타임 환산 가로선이 막대보다 높아도 세로축 안에 들어온다 (Astryx는 referenceLine을 범위에 넣지 않는다)", async () => {
    await openWar(RELIEVER, { ...HAND_WAR, rows: [{ ...W_RELIEVER, war_full: 3.0 }] });
    const chart = sectionOf(/시즌별 WAR/).charts[0]!;
    expect(Math.max(...tickValues(chart))).toBeGreaterThanOrEqual(3.0);
    expect(chart.textContent).toContain("풀타임 환산 3.0");
  });

  it("음수 WAR 시즌은 0 아래로 — 세로축이 음수까지 내려간다", async () => {
    const neg: WarRow = {
      ...W_RELIEVER,
      war_pit: -0.2,
      war: -0.2,
      war_full: -0.3,
      war_full_pit: -0.3,
      history: [
        { season: 2025, war_bat: null, war_pit: -0.85, war_pit_ra9: -1.2 },
        { season: 2026, war_bat: null, war_pit: 0.4, war_pit_ra9: 0.1 },
      ],
    };
    await openWar(RELIEVER, { ...HAND_WAR, rows: [neg] });
    const chart = sectionOf(/시즌별 WAR/).charts[0]!;
    expect(Math.min(...tickValues(chart))).toBeLessThanOrEqual(-0.85);
    expect(seasonSlices(describedText(chart), [2025, 2026]).get(2025)).toContain("−0.8"); // fmtWar(−0.85)
  });

  it("불펜 투수: 시즌별 투수 WAR 차트", async () => {
    await openWar(RELIEVER);
    const chart = sectionOf(/시즌별 WAR/).charts[0];
    expect(chart).toBeDefined();
    const slices = seasonSlices(describedText(chart!), [2025, 2026]);
    expect(slices.get(2025)).toContain("0.4"); // fmtWar(0.36)
    expect(slices.get(2026)).toContain("1.1"); // fmtWar(1.07)
  });
});

describe("2026 타자 WAR 구성 항목 (E29)", () => {
  it("제목은 2026 타자 WAR(fmtWar2), 6줄(이름 · 설명 · 점), 합계 '30.3점 → 3.14승', 그 시즌 리그 점 → 승 비율, 원본 그대로 안내", async () => {
    await openWar(TWO_WAY);
    const section = sectionOf(/2026 타자 WAR 3\.14는 이렇게 나왔다/);
    const values = ["+22.4", "−1.6", "0.0", "−7.5", "+0.9", "+16.1"]; // fmtRuns — 수비 0.04는 부호 없이 0.0
    let last = -1;
    BAT_RUNS_PARTS.forEach((part, i) => {
      const re = new RegExp(`${esc(part.name)}\\s*${esc(part.note)}\\s*${esc(values[i]!)}`);
      const at = section.text.search(re);
      expect(at, `${part.name} 줄이 없거나 차례가 다릅니다: ${section.text}`).toBeGreaterThan(last);
      last = at;
    });
    expect(section.text).toMatch(/합계/);
    expect(section.text).toMatch(/30\.3점\s*→\s*3\.14승/); // 구성 합(부호 없는 1자리) → 2026 war_bat(fmtWar2)
    expect(section.text).toContain(runsPerWinNote({ season: 2026, value: 10.37 })); // "2026년은 약 10.4점이 1승"
    expect(section.text).toContain(RUNS_NOT_EXACT);
    expect(section.text).not.toMatch(/\+30\.3/); // 합계에는 + 부호를 붙이지 않는다
  });

  it("합성 war.json 그대로: 제목 3.69 · 합계 36.2점 → 3.69승 · 2026년은 약 9.8점이 1승", async () => {
    const twoWay1: PlayerDetail = { ...TWO_WAY, player: { ...TWO_WAY.player, player_id: 1, name: "Sample Batter" } };
    await openWar(twoWay1, SAMPLE_WAR);
    const section = sectionOf(/2026 타자 WAR 3\.69는 이렇게 나왔다/);
    for (const v of ["+29.2", "+1.4", "0.0", "−14.5", "+1.3", "+18.8"]) expect(section.text).toContain(v);
    expect(section.text).toMatch(/36\.2점\s*→\s*3\.69승/);
    expect(section.text).toContain("2026년은 약 9.8점이 1승");
  });

  it("타자만: 제목 4.18은(끝자리 '팔'에 받침 — 조사 은) · 합계 45.0점 → 4.18승", async () => {
    await openWar(BATTER);
    const section = sectionOf(/2026 타자 WAR 4\.18은 이렇게 나왔다/);
    for (const v of ["+18.3", "+2.6", "−3.2", "+7.5", "+0.4", "+19.4"]) expect(section.text).toContain(v);
    expect(section.text).toMatch(/45\.0점\s*→\s*4\.18승/);
    expect(screen.queryByRole("heading", { name: /4\.18는/ })).toBeNull();
  });

  it("합계는 구성 항목 표의 마지막 줄이다 (시안 SA-7) — 항목 · 점→승 비율 · 합계", async () => {
    await openWar(BATTER);
    const section = sectionOf(/이렇게 나왔다/);
    expect(section.tables).toHaveLength(1);
    const grid = readTable(section.tables[0]!);
    expect(grid.rows).toHaveLength(BAT_RUNS_PARTS.length + 1);
    const last = grid.rows[grid.rows.length - 1]!;
    expect(last[0]).toBe("합계");
    expect(last[1]).toBe(runsPerWinNote({ season: 2026, value: 10.37 }));
    expect(last[2]).toMatch(/^45\.0점\s*→\s*4\.18승$/);
  });

  it("구성 합이 음수면 − (U+2212)만 붙고 + 는 없다", async () => {
    const minus: WarRow = {
      ...W_BATTER,
      prev_bat_runs: {
        runs_batting: -20.1,
        runs_baserunning: -1.0,
        runs_fielding: -4.0,
        runs_positional: 2.5,
        runs_league: 0.2,
        runs_replacement: 19.4,
      },
      history: [{ season: 2026, war_bat: -0.31, war_pit: null, war_pit_ra9: null }],
    };
    await openWar(BATTER, { ...HAND_WAR, rows: [minus] });
    const section = sectionOf(/2026 타자 WAR −0\.31은 이렇게 나왔다/);
    expect(section.text).toMatch(/−3\.0점\s*→\s*−0\.31승/);
    expect(section.text).not.toMatch(/-3\.0|\+−/);
  });

  it("타자 구성 항목이 없으면(투수만) 이 구역이 없다", async () => {
    await openWar(RELIEVER);
    expect(screen.queryByRole("heading", { name: /이렇게 나왔다/ })).toBeNull();
    expect(mainText()).not.toContain(RUNS_NOT_EXACT);
  });
});

describe("투수 WAR — FIP 기반 (D40)", () => {
  /** 구역 안에서 FIP 기반 열이 있는 표 */
  function pitGrid(): Grid {
    const section = sectionOf(PIT_WAR_TITLE);
    const grid = section.tables.map(readTable).find((g) => g.headers.includes(PIT_WAR_FIP_COL));
    if (grid === undefined) throw new Error(`'${PIT_WAR_FIP_COL}' 열이 있는 표가 없습니다: ${section.text.slice(0, 200)}`);
    return grid;
  }

  it("투타 겸업: 시즌 · FIP 기반 · 실점 기반 표(fmtWar2, 없으면 —), 오래된 순, 아래 설명", async () => {
    await openWar(TWO_WAY);
    const grid = pitGrid();
    expect(grid.headers.slice(0, 3)).toEqual(["시즌", PIT_WAR_FIP_COL, PIT_WAR_RA9_COL]);
    expect(grid.rows.map((row) => (row[col(grid, "시즌")] ?? "").slice(0, 4))).toEqual(["2022", "2023", "2024", "2025", "2026"]);
    expect(cell(grid, 2022, PIT_WAR_FIP_COL)).toBe("3.47");
    expect(cell(grid, 2022, PIT_WAR_RA9_COL)).toBe("2.91");
    expect(cell(grid, 2023, PIT_WAR_FIP_COL)).toBe("—"); // 그 시즌 투수 기록 없음
    expect(cell(grid, 2023, PIT_WAR_RA9_COL)).toBe("—");
    expect(cell(grid, 2024, PIT_WAR_FIP_COL)).toBe("0.48");
    expect(cell(grid, 2024, PIT_WAR_RA9_COL)).toBe("−0.31"); // U+2212
    expect(cell(grid, 2026, PIT_WAR_FIP_COL)).toBe("2.13");
    expect(cell(grid, 2026, PIT_WAR_RA9_COL)).toBe("2.71");
    expect(sectionOf(PIT_WAR_TITLE).text).toContain(PIT_WAR_NOTE);
  });

  it("불펜 투수: 2025 · 2026 두 줄", async () => {
    await openWar(RELIEVER);
    const grid = pitGrid();
    expect(cell(grid, 2025, PIT_WAR_FIP_COL)).toBe("0.36");
    expect(cell(grid, 2025, PIT_WAR_RA9_COL)).toBe("0.52");
    expect(cell(grid, 2026, PIT_WAR_FIP_COL)).toBe("1.07");
    expect(cell(grid, 2026, PIT_WAR_RA9_COL)).toBe("−0.18");
  });

  it("투수 WAR이 한 시즌도 없으면(타자만) 이 구역이 없다", async () => {
    await openWar(BATTER);
    expect(screen.queryByRole("heading", { name: PIT_WAR_TITLE })).toBeNull();
    expect(mainText()).not.toContain(PIT_WAR_NOTE);
  });
});

describe("공통·접근성 (D16, N10)", () => {
  it("고지·출처는 기존 바닥글 그대로 한 번씩", async () => {
    await openWar(TWO_WAY);
    expect(screen.getByText(/재미로 보는 예측/)).toBeInTheDocument();
    expect(screen.getByText(/MLB Stats API, Baseball Savant/)).toBeInTheDocument();
  });

  it("WAR 구역을 더해도 제목은 h1 하나에서 차례로 내려간다", async () => {
    await openWar(TWO_WAY);
    const levels = screen.getAllByRole("heading").map((h) => headingLevel(h) ?? 2);
    expect(levels[0]).toBe(1);
    expect(levels.filter((l) => l === 1)).toHaveLength(1);
    levels.forEach((level, i) => {
      if (i > 0) expect(level).toBeLessThanOrEqual(levels[i - 1]! + 1);
    });
    // WAR 구역은 h2 하나, 그 아래 세 소구역은 h3 (시안 SA-7)
    expect(screen.getByRole("heading", { level: 2, name: WAR_LABEL })).toBeInTheDocument();
    for (const name of [/시즌별 WAR/, /2026 타자 WAR 3\.14는 이렇게 나왔다/, PIT_WAR_TITLE]) {
      expect(screen.getByRole("heading", { level: 3, name })).toBeInTheDocument();
    }
  });

  it("투수 WAR 표는 구역 제목으로 이름이 붙고, 기본 이름 'Table'로 읽히지 않는다", async () => {
    await openWar(TWO_WAY);
    expect(screen.getByRole("table", { name: PIT_WAR_TITLE })).toBeInTheDocument();
    expect(screen.queryAllByRole("group", { name: "Table" })).toHaveLength(0);
  });

  const noPrev: WarFile = { ...HAND_WAR, rows: [{ ...W_RELIEVER, prev_war: null }] };
  for (const [label, detail, war] of [
    ["투타 겸업", TWO_WAY, HAND_WAR],
    ["타자", BATTER, HAND_WAR],
    ["불펜", RELIEVER, HAND_WAR],
    ["불펜 · prev_war 없음", RELIEVER, noPrev],
    ["2026 미출전", ABSENT, HAND_WAR],
  ] as const) {
    it(`빈 값이 NaN·null·undefined로 새지 않는다 (${label})`, async () => {
      await openWar(detail, war);
      expect(mainText()).not.toMatch(/NaN|undefined|null|Infinity/);
    });
  }
});

describe("소스 규칙 (계약 사용 · Astryx — web/.claude/CLAUDE.md)", () => {
  const APP_PLAYER = join(import.meta.dirname, "..", "app", "player");

  /** 레인 B 파일: WarSection.tsx + war/ 아래 */
  const laneFiles = (): string[] => {
    const files = [join(APP_PLAYER, "WarSection.tsx")];
    const sub = join(APP_PLAYER, "war");
    if (existsSync(sub)) {
      for (const name of readdirSync(sub, { recursive: true }).map(String)) {
        if (/\.(tsx|ts)$/.test(name)) files.push(join(sub, name));
      }
    }
    return files;
  };

  /** 주석을 뺀 소스 */
  const code = (path: string) =>
    readFileSync(path, "utf-8")
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/(^|[^:])\/\/.*$/gm, "$1");

  it("war.json은 useJson(loadWar…)으로 받고 findWarRow로 행을 찾는다 · 문구는 warText, 숫자는 format · 차트는 쌓은 막대", () => {
    const src = laneFiles().map(code).join("\n");
    expect(src).toMatch(/useJson\(\s*(?:\(\)\s*=>\s*)?loadWar\b/);
    expect(src).toMatch(/findWarRow\(/);
    expect(src).toMatch(/from\s+["']@\/src\/warText["']/);
    expect(src).toMatch(/from\s+["']@\/src\/format["']/);
    expect(src).toMatch(/from\s+["']@astryxdesign\/charts["']/);
    expect(src).toMatch(/\bbar\(/);
    expect(src).toMatch(/\bstack\s*:/);
  });

  it("PlayerDetail.tsx가 WAR 구역을 붙인다", () => {
    expect(code(join(APP_PLAYER, "PlayerDetail.tsx"))).toMatch(/WarSection/);
  });

  it("<div>·<span>·style={…}·hex 색·임의 Tailwind 값·.css 가져오기가 없다", () => {
    for (const file of laneFiles()) {
      const src = code(file);
      const rel = file.slice(APP_PLAYER.length + 1);
      expect(src, `${rel}: <div>`).not.toMatch(/<div[\s>]/);
      expect(src, `${rel}: <span>`).not.toMatch(/<span[\s>]/);
      expect(src, `${rel}: style={…}`).not.toMatch(/\bstyle=\{/);
      expect(src, `${rel}: hex 색`).not.toMatch(/#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b/);
      expect(src, `${rel}: 임의 Tailwind 값`).not.toMatch(/\b[a-z][a-z0-9-]*-\[[^\]]+\]/);
      expect(src, `${rel}: .css 가져오기`).not.toMatch(/import\s+["'][^"']+\.css["']/);
    }
  });
});
