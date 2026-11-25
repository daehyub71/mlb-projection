// 선수 상세 (F14, 시안 SA-2, 주소 /player/?id= — DESIGN DS1) 컴포넌트 테스트 — 레인 B Red
// 데이터는 fetch를 바꿔 끼워 합성 샘플(tests/fixtures/synthetic/web)과 손으로 만든 PlayerDetail로 준다. 네트워크는 쓰지 않는다.
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PlayerPage from "@/app/player/page";
import { Providers } from "@/app/providers";
import { DATA_BASE } from "@/src/data/load";
import type { BatSeason, FieldingSeason, PitDetail, PitSeason, PlayerDetail } from "@/src/data/types";
import { initials, playerPhotoUrl } from "@/src/photo";
import { BAND_NOTE, PLAYING_TIME_NOTE } from "@/src/site";

// ---------------------------------------------------------------------------
// 주소(?id=)와 브라우저 대역
// ---------------------------------------------------------------------------

// next/navigation의 useSearchParams를 테스트 주소로 바꾼다 (정적 export라 페이지가 브라우저에서 ?id=를 읽는다)
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

type Reply = PlayerDetail | { status: number } | "reject" | "pending";

/** 경로 → 응답. 모르는 경로는 404 */
function stubFetch(routes: Record<string, Reply>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
    const reply = routes[String(input)];
    if (reply === undefined) return new Response("not found", { status: 404 });
    if (reply === "reject") throw new TypeError("Failed to fetch");
    if (reply === "pending") return new Promise<Response>(() => undefined);
    if ("status" in reply) return new Response("error", { status: reply.status });
    return new Response(JSON.stringify(reply), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function playerPath(id: number): string {
  return `${DATA_BASE}/players/${id}.json`;
}

function openPage(search: string, routes: Record<string, Reply> = {}) {
  setSearch(search);
  const fetchMock = stubFetch(routes);
  const view = render(
    <Providers>
      <PlayerPage />
    </Providers>,
  );
  return { fetchMock, ...view };
}

/** 선수 상세를 열고 이름(h1)이 뜰 때까지 기다린다 */
async function openPlayer(detail: PlayerDetail) {
  const opened = openPage(`?id=${detail.player.player_id}`, { [playerPath(detail.player.player_id)]: detail });
  await screen.findByRole("heading", { level: 1, name: (name) => name.startsWith(detail.player.name) });
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
// 화면 읽기 도우미 — 제목 기준 '구역'(화면 낭독기가 제목으로 건너뛰는 단위)과 표
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

/** start 뒤(start 안쪽 제외)이고 end 앞인 노드인가 */
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
  return out.replace(/\s+/g, " ");
}

function main(): HTMLElement {
  return screen.getByRole("main");
}

function mainText(): string {
  return clean(main().textContent);
}

interface Section {
  text: string;
  tables: HTMLTableElement[];
  links: HTMLAnchorElement[];
}

/** 제목부터 같은 수준 이하의 다음 제목 전까지 */
function sectionOf(name: RegExp): Section {
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
  return { text: textBetween(root, heading, stop), tables: pick("table"), links: pick("a") };
}

/** 머리 띠: 본문 처음부터 첫 h2 전까지 (사진·이름·한 줄 정보·칩) */
function bandText(): string {
  const firstH2 = Array.from(document.body.querySelectorAll("*")).find((el) => headingLevel(el) === 2);
  return clean(textBetween(document.body, screen.getByRole("navigation", { name: "주요 메뉴" }), firstH2 ?? document.querySelector("footer")));
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

/** 구역 안에서 '시즌' 열이 있는 표 (차트가 숨겨 두는 자료 표는 열 이름이 데이터 키라 걸리지 않는다) */
function seasonGrid(section: Section): Grid {
  const grid = section.tables.map(readTable).find((g) => g.headers.includes("시즌"));
  if (grid === undefined) throw new Error(`'시즌' 열이 있는 표가 없습니다: ${section.text.slice(0, 200)}`);
  return grid;
}

function col(grid: Grid, header: RegExp): number {
  const index = grid.headers.findIndex((h) => header.test(h));
  if (index < 0) throw new Error(`${header} 열이 없습니다: ${grid.headers.join(" | ")}`);
  return index;
}

function rowsOf(grid: Grid, season: number): string[][] {
  const seasonCol = col(grid, /^시즌$/);
  return grid.rows.filter((row) => (row[seasonCol] ?? "").startsWith(String(season)));
}

function cell(grid: Grid, season: number, header: RegExp): string {
  const row = rowsOf(grid, season)[0];
  if (row === undefined) throw new Error(`${season} 행이 없습니다`);
  return row[col(grid, header)] ?? "";
}

/** 표 숫자 → 수 (U+2212 빼기, + 부호, 천 단위 쉼표 허용). 숫자가 아니면 NaN */
function num(text: string): number {
  const body = text.replace(/−/g, "-").replace(/^\+/, "").replace(/,/g, "");
  return /^-?\d+(\.\d+)?$/.test(body) ? Number(body) : Number.NaN;
}

/** 머리글이 순서대로 들어 있는가 (사이에 다른 열은 허용 — SPEC D14의 평균 타구 속도 등) */
function expectHeadersInOrder(headers: string[], expected: RegExp[]): void {
  let next = 0;
  for (const header of headers) {
    if (next < expected.length && expected[next]!.test(header)) next += 1;
  }
  expect(next, `머리글 ${headers.join(" | ")}에 ${expected[next]}이(가) 순서대로 없습니다`).toBe(expected.length);
}

/** 역할 전환 단추 — SegmentedControl(radio)·TabList(button)·탭(tab) 어느 것이든 */
function roleControl(name: string): HTMLElement | null {
  return (
    screen.queryByRole("radio", { name }) ??
    screen.queryByRole("tab", { name }) ??
    screen.queryByRole("button", { name })
  );
}

function isSelected(el: HTMLElement): boolean {
  return ["aria-checked", "aria-selected", "aria-current", "aria-pressed"].some((attr) => {
    const value = el.getAttribute(attr);
    return value === "true" || value === "page";
  });
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

function hasLoadingIndicator(): boolean {
  return (
    screen.queryByText(/불러오는 중/) !== null ||
    screen.queryByRole("progressbar") !== null ||
    screen.queryByRole("status") !== null
  );
}

function linksHome(): boolean {
  return within(main())
    .queryAllByRole("link")
    .some((a) => a.getAttribute("href") === "/");
}

// ---------------------------------------------------------------------------
// 픽스처 — 합성 샘플 + 손으로 만든 PlayerDetail (타입을 그대로 지킨다)
// ---------------------------------------------------------------------------

const ROOT = join(import.meta.dirname, "..", "..");

function sample(name: string): PlayerDetail {
  return JSON.parse(readFileSync(join(ROOT, "tests/fixtures/synthetic/web", name), "utf-8")) as PlayerDetail;
}

/** 합성 타자: 2020(60경기)·2026 두 시즌, SS, 우타/우투, 현역 */
const SAMPLE_BAT = sample("player_bat.json");
/** 합성 투수: 현역 아님, 마지막 2025, 생일·팀 없음, 2026 미출전(prev null), ERA 범위 없음 */
const SAMPLE_PIT = sample("player_pit.json");

function batSeason(season: number, over: Partial<BatSeason> = {}): BatSeason {
  return {
    season,
    age: season - 1998,
    team_name: "Sample Team",
    is_short_season: season === 2020,
    pa: 600,
    hr: 25,
    obp: 0.34,
    slg: 0.46,
    ops: 0.8,
    k_pct: 22.0,
    bb_pct: 9.0,
    woba: 0.34,
    xwoba: 0.345,
    barrel_pct: 10.0,
    hard_hit_pct: 45.0,
    ev_avg: 90.5,
    ...over,
  };
}

/** 6시즌 — 표는 최근 5시즌(2022~2026)만 */
const BAT_HISTORY: BatSeason[] = [
  batSeason(2021, { ops: 0.7, team_name: "Old Team" }),
  batSeason(2022, { ops: 0.801, woba: null }),
  batSeason(2023, { ops: 0.842 }),
  batSeason(2024, { ops: 0.777 }),
  batSeason(2025, { ops: 0.815 }),
  batSeason(2026, {
    pa: 655,
    hr: 31,
    obp: 0.362,
    slg: 0.499,
    ops: 0.861,
    k_pct: 21.4,
    bb_pct: 10.2,
    woba: 0.368,
    xwoba: 0.371,
    barrel_pct: 12.9,
    hard_hit_pct: 48.1,
  }),
];

function fielding(season: number, over: Partial<FieldingSeason>): FieldingSeason {
  return {
    season,
    oaa: null,
    frv_total: null,
    frv_range: null,
    frv_arm: null,
    frv_dp: null,
    frv_framing: null,
    frv_blocking: null,
    frv_throwing: null,
    outs_by_pos: {},
    ...over,
  };
}

/** 유격수 2시즌. OAA+FRV(=15)는 어디에도 나오면 안 된다 (D21) */
const SS_FIELDING: FieldingSeason[] = [
  fielding(2025, { oaa: -2, frv_total: -1, frv_range: -1, frv_arm: 0, frv_dp: 0, outs_by_pos: { SS: 3100 } }),
  fielding(2026, { oaa: 6, frv_total: 9, frv_range: 5, frv_arm: 3, frv_dp: 1, outs_by_pos: { "2B": 120, SS: 3450 } }),
];

/** 타자 전용 (id 1): 예측·직전 시즌·그룹 오차는 합성 샘플 값, 기록은 6시즌 */
const BATTER: PlayerDetail = {
  meta: SAMPLE_BAT.meta,
  player: { ...SAMPLE_BAT.player },
  bat: { ...SAMPLE_BAT.bat!, history: BAT_HISTORY },
  pit: null,
  fielding: SS_FIELDING,
};

function pitSeason(season: number, over: Partial<PitSeason>): PitSeason {
  return {
    season,
    age: season - 1995,
    team_name: "Sample Team",
    is_short_season: false,
    g: 30,
    gs: 30,
    ip: 150,
    era: 3.5,
    fip: 3.6,
    k_pct: 25.0,
    bb_pct: 8.0,
    k_bb_pct: 17.0,
    xera: 3.6,
    xwoba_against: 0.3,
    whiff_pct: 28.0,
    ...over,
  };
}

/** 완전한 투수 예측 (투타 겸업용) — ERA·K−BB% 범위 모두 있음 */
const PIT_FULL: PitDetail = {
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
    pitSeason(2025, { g: 14, gs: 14, ip: 47.0, era: 2.87, fip: 3.1, k_pct: 30.2, bb_pct: 6.5, k_bb_pct: 23.7 }),
    pitSeason(2026, { g: 28, gs: 28, ip: 150.667, era: 3.1, fip: 3.25, k_pct: 27.5, bb_pct: 8.0, k_bb_pct: 19.5 }),
  ],
};

/** 투타 겸업 (id 3, F12) */
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
  bat: BATTER.bat,
  pit: PIT_FULL,
  fielding: [],
};

/** 포수 (id 4): 프레이밍·블로킹·송구 세부 */
const CATCHER: PlayerDetail = {
  ...BATTER,
  player: { ...BATTER.player, player_id: 4, name: "Sample Catcher", primary_position: "C" },
  fielding: [
    fielding(2026, { frv_total: 9, frv_framing: 7, frv_blocking: -2, frv_throwing: 4, outs_by_pos: { C: 8000 } }),
  ],
};

/** 지명타자 (id 5): 수비 기록 없음 */
const DH_ONLY: PlayerDetail = {
  ...BATTER,
  player: { ...BATTER.player, player_id: 5, name: "Sample Hitter", primary_position: "DH" },
  fielding: [],
};

// ---------------------------------------------------------------------------
// 테스트
// ---------------------------------------------------------------------------

describe("주소 ?id=와 불러오기 상태", () => {
  it("useSearchParams와 parsePlayerId를 Suspense 경계 안에서 쓴다 (정적 export)", () => {
    const dir = join(import.meta.dirname, "..", "app", "player");
    const source = readdirSync(dir, { recursive: true })
      .map(String)
      .filter((file) => /\.tsx?$/.test(file))
      .map((file) => readFileSync(join(dir, file), "utf-8"))
      .join("\n");
    expect(source).toMatch(/useSearchParams/);
    expect(source).toMatch(/parsePlayerId/);
    expect(source).toMatch(/<Suspense[\s>]/);
  });

  it("?id=의 선수 JSON을 먼저 받고, 그 뒤 WAR 구역용 war.json 하나만 더 받는다 (v0.5 F22)", async () => {
    const { fetchMock } = await openPlayer(BATTER);
    await waitFor(() =>
      expect(fetchMock.mock.calls.map(([url]) => String(url))).toEqual(["/data/players/1.json", "/data/war.json"]),
    );
  });

  it("받는 동안 불러오는 중임을 보여 준다", async () => {
    openPage("?id=1", { [playerPath(1)]: "pending" });
    await waitFor(() => expect(hasLoadingIndicator()).toBe(true));
  });

  for (const bad of ["?id=abc", "?id=0", "?id=1.5", "?id=-3"]) {
    it(`잘못된 ID(${bad})는 받지 않고 안내와 예측표 링크를 보여 준다`, async () => {
      const { fetchMock } = openPage(bad);
      await waitFor(() => expect(mainText()).toMatch(/올바르지 않/));
      expect(linksHome()).toBe(true);
      expect(fetchMock).not.toHaveBeenCalled();
    });
  }

  it("ID가 없으면 받지 않고 안내와 예측표 링크를 보여 준다", async () => {
    const { fetchMock } = openPage("");
    await waitFor(() => expect(mainText()).toMatch(/선수 ID/));
    expect(linksHome()).toBe(true);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("없는 선수(404)는 '찾을 수 없음'과 예측표 링크", async () => {
    openPage("?id=999", {});
    await waitFor(() => expect(mainText()).toMatch(/찾을 수 없/));
    expect(linksHome()).toBe(true);
  });

  it("서버 오류(500)는 404와 다른 '불러오지 못함' 안내", async () => {
    openPage("?id=1", { [playerPath(1)]: { status: 500 } });
    await waitFor(() => expect(mainText()).toMatch(/불러오지 못/));
    expect(mainText()).not.toMatch(/찾을 수 없/);
    expect(linksHome()).toBe(true);
  });

  it("네트워크 오류도 '불러오지 못함' 안내와 예측표 링크", async () => {
    openPage("?id=1", { [playerPath(1)]: "reject" });
    await waitFor(() => expect(mainText()).toMatch(/불러오지 못/));
    expect(linksHome()).toBe(true);
  });

  it("타자·투수 예측이 모두 없는 JSON은 안내와 예측표 링크 (F12: 기록 없는 역할은 예측하지 않는다)", async () => {
    const noRole: PlayerDetail = { ...BATTER, player: { ...BATTER.player, player_id: 6, name: "Nobody Sample" }, bat: null, pit: null };
    openPage("?id=6", { [playerPath(6)]: noRole });
    await waitFor(() => expect(mainText()).toMatch(/Nobody Sample의 2027 예측이 없습니다/));
    expect(linksHome()).toBe(true);
    expect(mainText()).not.toMatch(/NaN|undefined|null|Infinity/);
  });

  it("오류 화면에도 공통 머리글·바닥글(고지·출처)이 있다", async () => {
    openPage("?id=999", {});
    await waitFor(() => expect(mainText()).toMatch(/찾을 수 없/));
    expect(screen.getByRole("navigation", { name: "주요 메뉴" })).toBeInTheDocument();
    expect(screen.getByText(/재미로 보는 예측/)).toBeInTheDocument();
    expect(screen.getByText(/MLB Stats API, Baseball Savant/)).toBeInTheDocument();
  });
});

describe("머리 띠 (SA-2)", () => {
  it("이름(h1)·투타·2027 나이·최근 팀·마지막 MLB 시즌·사용한 기록 칩", async () => {
    await openPlayer(BATTER);
    const band = bandText();
    expect(band).toMatch(/우타\/우투/); // bats R · throws R
    expect(band).toMatch(/2027년\s*29세/); // seasonAge("1998-04-01", 2027)
    expect(band).toContain("Sample Team"); // 마지막 기록 행(2026)의 팀
    expect(band).not.toContain("Old Team");
    expect(band).toMatch(/마지막 MLB 시즌\s*2026/);
    expect(band).toMatch(/사용한 기록\s*2022~2026\s*·\s*5시즌\s*·\s*3,012타석/);
  });

  it("현역이면 아무 표시도 하지 않는다 (D23)", async () => {
    await openPlayer(BATTER);
    expect(mainText()).not.toMatch(/현역/);
  });

  it("현역 아님 · 마지막 MLB 시즌 · 투수 표본 (D23, 합성 투수)", async () => {
    await openPlayer(SAMPLE_PIT);
    const band = bandText();
    expect(band).toMatch(/현역 아님/);
    expect(band).toMatch(/마지막 MLB 시즌\s*2025/);
    expect(band).toMatch(/좌타\/좌투/);
    expect(band).toMatch(/사용한 기록\s*2024~2025\s*·\s*2시즌\s*·\s*(402|95\.2)/); // 상대 타자 402 또는 95.2이닝
    expect(band).not.toMatch(/2027년\s*\S*세/); // 생일 없음 → 나이 없음
  });

  it("투타 겸업은 포지션을 '투타 겸업'으로", async () => {
    await openPlayer(TWO_WAY);
    const band = bandText();
    expect(band).toMatch(/투타 겸업/);
    expect(band).toMatch(/좌타\/우투/);
    expect(band).toMatch(/2027년\s*32세/);
  });

  it("MLB 공식 사진과 그 가까이 '선수 사진 © MLB' (D27)", async () => {
    const { container } = await openPlayer(BATTER);
    const img = container.querySelector<HTMLImageElement>('img[src*="mlbstatic.com"]');
    expect(img).not.toBeNull();
    expect(img!.getAttribute("src")).toContain("/people/1/headshot/");
    expect(img!.getAttribute("src")).toMatch(/^https:\/\/img\.mlbstatic\.com\//);
    expect(playerPhotoUrl(1)).toContain("/people/1/headshot/");
    expect(screen.getAllByRole("img", { name: /Sample Batter/ }).length).toBeGreaterThan(0); // 대체 글
    expect(bandText()).toMatch(/선수 사진 © MLB/); // 바닥글 말고 머리 띠에도
  });

  it("사진을 불러오지 못하면 이니셜 배지로", async () => {
    const { container } = await openPlayer(BATTER);
    fireEvent.error(container.querySelector('img[src*="mlbstatic.com"]')!);
    await waitFor(() => expect(container.querySelector('img[src*="mlbstatic.com"]')).toBeNull());
    // Astryx Avatar는 이니셜을 CSS로 그리고 이름을 aria-label로 단다 — 직접 만든 배지면 글자 "SB"
    const badge =
      screen.queryAllByText(initials("Sample Batter")).length + screen.queryAllByRole("img", { name: /Sample Batter/ }).length;
    expect(badge).toBeGreaterThan(0);
  });

  it("사진 설정을 끄면 MLB 서버를 부르지 않고 저작권 표기도 없다", async () => {
    vi.stubEnv("NEXT_PUBLIC_PLAYER_PHOTOS", "off"); // photosEnabled()는 렌더 때 부른다
    const { container } = await openPlayer(BATTER);
    expect(container.querySelector('img[src*="mlbstatic.com"]')).toBeNull();
    expect(screen.queryByText(/선수 사진 © MLB/)).toBeNull();
    const badge =
      screen.queryAllByText(initials("Sample Batter")).length + screen.queryAllByRole("img", { name: /Sample Batter/ }).length;
    expect(badge).toBeGreaterThan(0);
  });
});

describe("역할 전환 (F12)", () => {
  it("타자만 있으면 전환 없이 타자 예측", async () => {
    await openPlayer(BATTER);
    expect(roleControl("투수")).toBeNull();
    expect(mainText()).toMatch(/2027 OPS/);
    expect(mainText()).not.toMatch(/2027 ERA/);
  });

  it("투수만 있으면 전환 없이 투수 예측", async () => {
    await openPlayer(SAMPLE_PIT);
    expect(roleControl("타자")).toBeNull();
    expect(mainText()).toMatch(/2027 ERA/);
    expect(mainText()).not.toMatch(/2027 OPS/);
  });

  it("투타 겸업은 타자가 기본이고 '투수'를 누르면 카드·오차·차트·표가 투수로 바뀐다", async () => {
    await openPlayer(TWO_WAY);
    const bat = roleControl("타자");
    const pit = roleControl("투수");
    expect(bat).not.toBeNull();
    expect(pit).not.toBeNull();
    expect(isSelected(bat!)).toBe(true);
    expect(mainText()).toMatch(/2027 OPS/);

    fireEvent.click(pit!);
    await waitFor(() => expect(mainText()).toMatch(/2027 ERA/));
    expect(isSelected(roleControl("투수")!)).toBe(true);
    expect(mainText()).not.toMatch(/2027 OPS/);
    expect(mainText()).toMatch(/2027 ERA\s*3\.42\s*±1\.38/);
    expect(sectionOf(/얼마나 빗나갈까/).text).toContain(".843"); // ERA MAE
    expect(screen.getByRole("img", { name: /ERA/ })).toBeInTheDocument();
    expectHeadersInOrder(seasonGrid(sectionOf(/최근 5시즌/)).headers, [/^G$/, /^GS$/, /^IP$/, /^ERA$/]);

    fireEvent.click(roleControl("타자")!);
    await waitFor(() => expect(mainText()).toMatch(/2027 OPS/));
  });
});

describe("2027 예측 카드", () => {
  it("타자: OPS ±범위·2026 실제·리그 평균 차이, PA*·HR*", async () => {
    await openPlayer(BATTER);
    const text = mainText();
    expect(text).toMatch(/2027 OPS\s*\.823\s*±\.097/); // fmtRate · fmtBand(band80.ops)
    expect(text).toMatch(/2026 실제\s*\.861/);
    expect(text).toMatch(/리그 평균\s*\.718[^\d+−]*\+\.105/); // fmtDiff(.823 − .718)
    expect(text).toMatch(/2027 PA\s*\*\s*610/); // fmtCount(610.4)
    expect(text).toMatch(/2026 실제\s*655/);
    expect(text).toMatch(/2027 HR\s*\*\s*27/); // fmtCount(27.1)
    expect(text).toMatch(/2026 실제\s*31/);
  });

  it("투수: ERA ±범위, IP*(야구 이닝 표기), K−BB% ±범위, 리그 평균 차이 (투타 겸업 → 투수)", async () => {
    await openPlayer(TWO_WAY);
    fireEvent.click(roleControl("투수")!);
    await waitFor(() => expect(mainText()).toMatch(/2027 ERA/));
    const text = mainText();
    expect(text).toMatch(/2027 ERA\s*3\.42\s*±1\.38/);
    expect(text).toMatch(/2026 실제\s*3\.10/);
    expect(text).toMatch(/리그 평균\s*4\.15[^\d+−]*−0\.73/);
    expect(text).toMatch(/2027 IP\s*\*\s*160\.1/); // fmtIp(160.3) = 481아웃
    expect(text).toMatch(/2026 실제\s*150\.2/); // fmtIp(150.667)
    expect(text).toMatch(/2027 K[−-]BB%\s*19\.6\s*±5\.5/);
    expect(text).toMatch(/2026 실제\s*19\.5/);
    expect(text).toMatch(/리그 평균\s*13\.6[^\d+−]*\+6\.0/);
  });

  it("범위가 없으면 ±를 붙이지 않고, 2026 미출전이면 '2026 실제'가 없다 (합성 투수)", async () => {
    await openPlayer(SAMPLE_PIT);
    const text = mainText();
    expect(text).toMatch(/2027 ERA\s*4\.05/);
    expect(text).not.toMatch(/2027 ERA\s*4\.05\s*±/); // band80.era = null
    expect(text).toMatch(/2027 IP\s*\*\s*41\.1/); // fmtIp(41.3)
    expect(text).toMatch(/2027 K[−-]BB%\s*14\.2\s*±5\.9/);
    expect(text).toMatch(/리그 평균\s*4\.15[^\d+−]*−0\.10/);
    expect(text).toMatch(/리그 평균\s*13\.6[^\d+−]*\+0\.6/);
    expect(text).not.toMatch(/2026 실제/); // prev = null
  });
});

describe("오차 안내 (그룹 과거 오차·출전량)", () => {
  it("'이 예측은 얼마나 빗나갈까' — 그룹 이름, MAE(fmtMae), 80% 범위(fmtBand), 검증 링크", async () => {
    await openPlayer(BATTER);
    const box = sectionOf(/이 예측은 얼마나 빗나갈까/);
    expect(box.text).toContain("3~5시즌"); // CAREER_LABEL.seasons_3_5
    expect(box.text).toContain("±.097");
    expect(box.text).toContain(".0660"); // fmtMae("ops", 0.066)
    expect(box.text).toContain("131.0"); // fmtMae("pa", 131)
    expect(box.text).toContain("7.20"); // fmtMae("hr", 7.2)
    // next/link는 테스트에서 끝 빗금을 떼고(trailingSlash는 빌드에서만 적용) 빌드에서는 붙인다
    expect(box.links.some((a) => /^\/validation\/?$/.test(a.getAttribute("href") ?? ""))).toBe(true);
  });

  it("투수 1~2시즌 그룹 (합성 투수)", async () => {
    await openPlayer(SAMPLE_PIT);
    const box = sectionOf(/이 예측은 얼마나 빗나갈까/);
    expect(box.text).toContain("1~2시즌");
    expect(box.text).toContain("±5.9");
    expect(box.text).toContain("29.5"); // fmtMae("ip", 29.5)
    expect(box.text).toContain("3.80"); // fmtMae("k_bb_pct", 3.8)
  });

  it("출전량 안내 상자 (D30)", async () => {
    await openPlayer(BATTER);
    expect(main().textContent).toContain(PLAYING_TIME_NOTE);
  });
});

describe("시즌별 추이 차트 (D26)", () => {
  it("타자: 시즌별 OPS 선 + 2027 점·±범위 + 리그 평균선, 값을 읽어 주는 설명", async () => {
    await openPlayer(BATTER);
    const chart = screen.getByRole("img", { name: /OPS/ });
    expect(chart.tagName.toLowerCase()).toBe("svg"); // @astryxdesign/charts의 Chart
    const desc = describedText(chart);
    for (const part of ["2022", ".801", "2026", ".861", "2027", ".823", "±.097", "리그 평균", ".718"]) {
      expect(desc, `차트 설명에 ${part}이(가) 없습니다: ${desc}`).toContain(part);
    }
    // 리그 평균선은 그림 안에 이름표로
    expect(Array.from(chart.querySelectorAll("text")).some((t) => t.textContent?.includes("리그 평균"))).toBe(true);
    // 가리키면 값 — charts 라이브러리의 툴팁 층. 툴팁은 svg가 붙은 뒤 effect에서 자리를 잡으므로 기다린다
    await waitFor(() => expect(chart.parentElement?.querySelector('[role="tooltip"]')).not.toBeNull());
    // 2027 점의 세로 막대 설명 (D29)
    expect(sectionOf(/시즌별 OPS/).text).toContain(BAND_NOTE);
  });

  it("투수: 시즌별 ERA (합성 투수)", async () => {
    await openPlayer(SAMPLE_PIT);
    const chart = screen.getByRole("img", { name: /ERA/ });
    const desc = describedText(chart);
    for (const part of ["2025", "3.79", "2027", "4.05", "리그 평균", "4.15"]) {
      expect(desc, `차트 설명에 ${part}이(가) 없습니다: ${desc}`).toContain(part);
    }
    // ERA 범위가 없으면(band80.era = null) 세로 막대 설명도 없다
    expect(sectionOf(/시즌별 ERA/).text).not.toContain(BAND_NOTE);
  });

  it("2026에 뛰지 않았으면 축에 빈 2026 칸을 두어 2025와 2027 예측이 붙어 보이지 않는다 (합성 투수)", async () => {
    await openPlayer(SAMPLE_PIT);
    const chart = screen.getByRole("img", { name: /ERA/ });
    const ticks = Array.from(chart.querySelectorAll("text")).map((t) => clean(t.textContent));
    const at = (label: string) => ticks.indexOf(label);
    expect(at("2025"), `축 이름표: ${ticks.join(" | ")}`).toBeGreaterThanOrEqual(0);
    expect(at("2026"), `축 이름표: ${ticks.join(" | ")}`).toBeGreaterThan(at("2025"));
    expect(at("2027 예측"), `축 이름표: ${ticks.join(" | ")}`).toBeGreaterThan(at("2026"));
    expect(describedText(chart)).not.toMatch(/2026 \d/); // 빈 칸에는 값이 없다
  });
});

describe("ERA 축 상한 (2026-11-25 사용자 결정)", () => {
  it("표본이 아주 작은 시즌의 ERA는 축을 늘리지 않고 위쪽 끝에 실제 값을 적는다", async () => {
    const base = SAMPLE_PIT.pit!;
    const first = base.history[0]!;
    const outlier: PlayerDetail = {
      ...SAMPLE_PIT,
      pit: { ...base, history: [{ ...first, season: 2024, ip: 0.3333, era: 108 }, ...base.history.slice(1)] },
    };
    await openPlayer(outlier);
    const chart = screen.getByRole("img", { name: /ERA/ });
    const texts = Array.from(chart.querySelectorAll("text")).map((t) => clean(t.textContent));
    expect(texts.some((t) => t.includes("↑ 108.00")), `차트 글자: ${texts.join(" | ")}`).toBe(true);
    // 축 눈금은 리그 평균(4.15)의 3배 언저리까지만 — 108까지 늘어나지 않는다
    const ticks = texts.filter((t) => /^\d+\.\d\d$/.test(t)).map(Number);
    expect(ticks.length).toBeGreaterThan(0);
    expect(Math.max(...ticks)).toBeLessThan(20);
    // 낭독 설명에는 실제 값이 그대로 남는다
    expect(describedText(chart)).toContain("108.00");
  });
});

describe("최근 5시즌 표 (D14)", () => {
  it("타자 열과 값, 최근 5시즌만 오래된 순으로", async () => {
    await openPlayer(BATTER);
    const grid = seasonGrid(sectionOf(/최근 5시즌/));
    expectHeadersInOrder(grid.headers, [
      /^시즌$/,
      /^팀$/,
      /^나이$/,
      /^PA$/,
      /^HR$/,
      /^OBP$/,
      /^SLG$/,
      /^OPS$/,
      /^wOBA$/,
      /^xwOBA$/,
      /^Barrel%$/,
      /^Hard.?Hit%$/i,
      /^K%\s*\/\s*BB%$/,
    ]);
    const seasonCol = col(grid, /^시즌$/);
    expect(grid.rows.map((row) => (row[seasonCol] ?? "").slice(0, 4))).toEqual(["2022", "2023", "2024", "2025", "2026"]);
    expect(cell(grid, 2026, /^팀$/)).toBe("Sample Team");
    expect(cell(grid, 2026, /^나이$/)).toBe("28");
    expect(cell(grid, 2026, /^PA$/)).toBe("655");
    expect(cell(grid, 2026, /^HR$/)).toBe("31");
    expect(cell(grid, 2026, /^OBP$/)).toBe(".362");
    expect(cell(grid, 2026, /^SLG$/)).toBe(".499");
    expect(cell(grid, 2026, /^OPS$/)).toBe(".861");
    expect(cell(grid, 2026, /^wOBA$/)).toBe(".368");
    expect(cell(grid, 2026, /^xwOBA$/)).toBe(".371");
    expect(cell(grid, 2026, /^Barrel%$/)).toBe("12.9");
    expect(cell(grid, 2026, /^Hard.?Hit%$/i)).toBe("48.1");
    expect(cell(grid, 2026, /^K%\s*\/\s*BB%$/)).toMatch(/^21\.4\s*\/\s*10\.2$/);
    expect(cell(grid, 2022, /^wOBA$/)).toBe("—"); // null
  });

  it("2020은 60경기 시즌으로 표시 (합성 타자)", async () => {
    await openPlayer(SAMPLE_BAT);
    const grid = seasonGrid(sectionOf(/최근 5시즌/));
    expect(rowsOf(grid, 2020).flat().join(" ")).toMatch(/60경기/);
    expect(rowsOf(grid, 2026).flat().join(" ")).not.toMatch(/60경기/);
  });

  it("투수 열과 값 (합성 투수)", async () => {
    await openPlayer(SAMPLE_PIT);
    const grid = seasonGrid(sectionOf(/최근 5시즌/));
    expectHeadersInOrder(grid.headers, [
      /^시즌$/,
      /^팀$/,
      /^나이$/,
      /^G$/,
      /^GS$/,
      /^IP$/,
      /^ERA$/,
      /^FIP$/,
      /^K%$/,
      /^BB%$/,
      /^K[−-]BB%$/,
      /^xERA$/,
      /^xwOBA 허용$/,
      /^Whiff%$/,
    ]);
    expect(cell(grid, 2025, /^팀$/)).toBe("—"); // team_name null
    expect(cell(grid, 2025, /^나이$/)).toBe("30");
    expect(cell(grid, 2025, /^G$/)).toBe("38");
    expect(cell(grid, 2025, /^GS$/)).toBe("0");
    expect(cell(grid, 2025, /^IP$/)).toBe("40.1"); // fmtIp(40.333)
    expect(cell(grid, 2025, /^ERA$/)).toBe("3.79");
    expect(cell(grid, 2025, /^FIP$/)).toBe("3.91");
    expect(cell(grid, 2025, /^K%$/)).toBe("24.0");
    expect(cell(grid, 2025, /^BB%$/)).toBe("9.1");
    expect(cell(grid, 2025, /^K[−-]BB%$/)).toBe("14.9");
    expect(cell(grid, 2025, /^xERA$/)).toBe("3.95");
    expect(cell(grid, 2025, /^xwOBA 허용$/)).toBe(".301");
    expect(cell(grid, 2025, /^Whiff%$/)).toBe("27.3");
  });
});

describe("수비 (D21, 과거 기록만)", () => {
  it("시즌별 포지션·아웃 수, OAA, FRV 합계와 세부 — OAA와 FRV를 더하지 않는다", async () => {
    await openPlayer(BATTER);
    const grid = seasonGrid(sectionOf(/수비/));
    const row2026 = rowsOf(grid, 2026).flat().join(" ");
    expect(row2026).toMatch(/SS[\s\S]*3,450/);
    expect(row2026).toMatch(/2B[\s\S]*120/);
    expect(row2026.indexOf("SS")).toBeLessThan(row2026.indexOf("2B")); // 아웃 많은 포지션부터
    expect(num(cell(grid, 2026, /OAA/))).toBe(6);
    expect(num(cell(grid, 2025, /OAA/))).toBe(-2);
    expect(grid.headers.some((h) => /FRV/.test(h))).toBe(true);
    const values2026 = rowsOf(grid, 2026).flat().map(num);
    for (const v of [9, 5, 3, 1]) expect(values2026, `FRV 값 ${v}이(가) 없습니다`).toContain(v);
    const everyValue = grid.rows.flat().map(num);
    expect(everyValue).not.toContain(15); // 6 + 9
    expect(everyValue).not.toContain(-3); // -2 + -1
  });

  it("포수는 프레이밍·블로킹·송구 세부", async () => {
    await openPlayer(CATCHER);
    const grid = seasonGrid(sectionOf(/수비/));
    expect(rowsOf(grid, 2026).flat().join(" ")).toContain("8,000");
    expect(num(cell(grid, 2026, /프레이밍|framing/i))).toBe(7);
    expect(num(cell(grid, 2026, /블로킹|blocking/i))).toBe(-2);
    expect(rowsOf(grid, 2026).flat().map(num)).toContain(4); // 송구(도루 저지)
  });

  it("수비 기록이 없으면 빈 상태 안내 (지명타자)", async () => {
    await openPlayer(DH_ONLY);
    const section = sectionOf(/수비/);
    expect(section.text).toMatch(/수비 기록이 없습니다/);
    expect(section.text).toMatch(/지명타자/);
    expect(section.tables.map(readTable).some((g) => g.headers.includes("시즌"))).toBe(false);
  });

  it("투수만 있는 선수는 투수 수비를 모으지 않는다고 알린다 (합성 투수)", async () => {
    await openPlayer(SAMPLE_PIT);
    const section = sectionOf(/수비/);
    expect(section.text).toMatch(/수비 기록이 없습니다/);
    expect(section.text).toMatch(/투수 수비는 모으지 않/);
    expect(section.text).not.toMatch(/수비수라면/);
    expect(section.tables.map(readTable).some((g) => g.headers.includes("시즌"))).toBe(false);
  });
});

describe("공통·접근성 (D16, N10)", () => {
  it("공통 머리글(예측이 현재 메뉴)과 바닥글", async () => {
    await openPlayer(BATTER);
    expect(screen.getByRole("navigation", { name: "주요 메뉴" })).toBeInTheDocument();
    expect(screen.getByRole("link", { current: "page" })).toHaveTextContent("예측");
    expect(screen.getByText(/재미로 보는 예측/)).toBeInTheDocument();
    expect(screen.getByText(/MLB Stats API, Baseball Savant/)).toBeInTheDocument();
  });

  it("제목은 h1 하나(선수 이름)에서 차례로 내려가고 구역 제목이 있다", async () => {
    await openPlayer(BATTER);
    const levels = screen.getAllByRole("heading").map((h) => headingLevel(h) ?? 2);
    expect(levels[0]).toBe(1);
    expect(levels.filter((l) => l === 1)).toHaveLength(1);
    levels.forEach((level, i) => {
      if (i > 0) expect(level).toBeLessThanOrEqual(levels[i - 1]! + 1);
    });
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Sample Batter");
    for (const name of [/이 예측은 얼마나 빗나갈까/, /최근 5시즌/, /수비/]) {
      expect(screen.getByRole("heading", { name })).toBeInTheDocument();
    }
  });

  it("기록 표는 구역 제목으로 이름이 붙고, 기본 이름 'Table'로 읽히지 않는다", async () => {
    await openPlayer(BATTER);
    expect(screen.getByRole("table", { name: "최근 5시즌 기록" })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "수비 (과거 기록만)" })).toBeInTheDocument();
    expect(screen.queryAllByRole("group", { name: "Table" })).toHaveLength(0);
  });

  for (const [label, detail] of [
    ["타자", BATTER],
    ["합성 투수", SAMPLE_PIT],
    ["투타 겸업", TWO_WAY],
  ] as const) {
    it(`빈 값이 NaN·null·undefined로 새지 않는다 (${label})`, async () => {
      await openPlayer(detail);
      expect(mainText()).not.toMatch(/NaN|undefined|null|Infinity/);
    });
  }
});
