// 검증 화면 WAR 구역 (SA-8, SPEC F22·D36·D39·D40·D41·E28·E29) — M9 레인 C Red 테스트
// fetch는 vi.stubGlobal로 합성 샘플·손으로 만든 war.json을 준다. 네트워크는 쓰지 않는다.
// 손으로 만든 값은 실제 2026 시험 값(E28)과 일부러 다르게 둔다 — 화면이 값을 데이터에서 읽는지 본다.
// 실존 선수 이름은 쓰지 않는다 (합성 샘플의 "Sample Batter"와 지어낸 이름만).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ValidationPage from "@/app/validation/page";
import { Providers } from "@/app/providers";
import type {
  Role,
  ValidationReport,
  WarBatRuns,
  WarEvalRow,
  WarFile,
  WarGroup,
  WarMetric,
  WarRoles,
  WarRow,
} from "@/src/data/types";
import { fmtCount, fmtRuns, fmtWar2, fmtWarMae } from "@/src/format";
import {
  BAT_RUNS_PARTS,
  BAT_WAR_SUMMARY,
  FULL_TIME_SHORT,
  PIT_WAR_STABILITY,
  PIT_WAR_TITLE,
  WAR_FACTS,
  WAR_TEST_INTRO,
  WAR_WHAT,
} from "@/src/warText";

const ROOT = join(import.meta.dirname, "..", "..");

function synthetic<T>(name: string): T {
  return JSON.parse(readFileSync(join(ROOT, "tests/fixtures/synthetic/web", name), "utf-8")) as T;
}

const syntheticValidation = synthetic<ValidationReport>("validation.json");
const syntheticWar = synthetic<WarFile>("war.json");

// ---------------------------------------------------------------------------
// 손으로 만든 war.json
// ---------------------------------------------------------------------------

/** 2026 시험 표의 여섯 칸 — 이 순서로 보여 준다 (SA-8) */
const CELLS: ReadonlyArray<{ role: Role; metric: WarMetric; group: WarGroup; what: string; who: string }> = [
  { role: "bat", metric: "war", group: "all", what: "타자 예측 WAR", who: "전원" },
  { role: "pit", metric: "war", group: "all", what: "투수 예측 WAR", who: "전원" },
  { role: "bat", metric: "war_full", group: "regulars", what: `타자 ${FULL_TIME_SHORT}`, who: "주전급" },
  { role: "pit", metric: "war_full", group: "regulars", what: `투수 ${FULL_TIME_SHORT}`, who: "주전급" },
  { role: "bat", metric: "war", group: "war2plus", what: "타자 예측 WAR", who: "실제 WAR 2 이상" },
  { role: "pit", metric: "war", group: "war2plus", what: "투수 예측 WAR", who: "실제 WAR 2 이상" },
];

/** 시즌 → 여섯 칸(CELLS 순서)의 [대상 수, 이 모델, 작년 그대로, 리그 평균 비율] */
type CellValues = [n: number, mae: number, prev: number, league: number];
const SEASON_VALUES: Record<number, CellValues[]> = {
  2026: [
    [1234, 0.4321, 0.5432, 0.4987],
    [1456, 0.3012, 0.3456, 0.3321],
    [131, 1.3876, 1.6912, 1.4123],
    [64, 1.0234, 1.1987, 1.4456],
    [117, 1.6543, 1.7789, 2.1234],
    [79, 1.7654, 1.5123, 2.3345], // 작년 그대로에 진다
  ],
  2025: [
    [1101, 0.4711, 0.5612, 0.5523],
    [1402, 0.2999, 0.3511, 0.3399],
    [140, 1.3011, 1.6122, 1.5433],
    [70, 0.8912, 1.0511, 1.2711],
    [120, 1.6611, 1.6422, 2.4111], // 작년 그대로에 진다
    [66, 1.6211, 1.4311, 2.2611], // 작년 그대로에 진다
  ],
  2024: [
    [1090, 0.5099, 0.5899, 0.5799],
    [1388, 0.3101, 0.3622, 0.3355],
    [125, 1.5511, 1.6411, 1.5011], // 리그 평균 비율에 진다
    [67, 0.9411, 1.0711, 1.2211],
    [126, 1.8811, 1.9122, 2.4411],
    [68, 1.6611, 1.4911, 1.5733], // 두 기준선에 진다
  ],
};

/** 경력 그룹 행 — 시험 표 여섯 칸에는 들어가지 않는다 (선수 상세의 그룹 평균 오차용, D39) */
const GROUP_ONLY_MAE = 0.2777;

function evalRow(
  target_season: number,
  role: Role,
  metric: WarMetric,
  group: WarGroup,
  [n, mae, prev, league]: CellValues,
): WarEvalRow {
  return { target_season, role, metric, group, n, mae, baseline_mae_prev: prev, baseline_mae_league: league };
}

function evaluations(): WarEvalRow[] {
  const out: WarEvalRow[] = [];
  for (const season of [2024, 2025, 2026]) {
    SEASON_VALUES[season]!.forEach((values, i) => {
      const cell = CELLS[i]!;
      out.push(evalRow(season, cell.role, cell.metric, cell.group, values));
    });
    out.push(evalRow(season, "bat", "war", "seasons_1_2", [555, GROUP_ONLY_MAE, 0.3111, 0.3011]));
    out.push(evalRow(season, "pit", "war", "seasons_3_5", [744, 0.4444, 0.5111, 0.4888]));
  }
  return out;
}

function warRow(
  id: number,
  name: string,
  roles: WarRoles,
  pred: { bat: number | null; pit: number | null },
  last: { bat: number | null; pit: number | null; runs: WarBatRuns | null },
): WarRow {
  const sum = (a: number | null, b: number | null) => (a ?? 0) + (b ?? 0);
  return {
    player: {
      player_id: id,
      name,
      primary_position: roles === "pit" ? "P" : roles === "both" ? "TWP" : "SS",
      age: 28,
      team_name: "Sample Team",
      active: true,
      last_mlb_season: 2026,
    },
    roles,
    played_prev: true,
    group: "seasons_3_5",
    slot: roles === "bat" ? null : "sp",
    pa: roles === "pit" ? null : 520.4,
    ip: roles === "bat" ? null : 120.3,
    war_bat: pred.bat,
    war_pit: pred.pit,
    war: sum(pred.bat, pred.pit),
    war_full_bat: pred.bat === null ? null : pred.bat * 1.2,
    war_full_pit: pred.pit === null ? null : pred.pit * 1.3,
    war_full: sum(pred.bat === null ? null : pred.bat * 1.2, pred.pit === null ? null : pred.pit * 1.3),
    prev_war: sum(last.bat, last.pit),
    prev_bat_runs: last.runs,
    history: [
      { season: 2025, war_bat: last.bat === null ? null : 1.11, war_pit: last.pit === null ? null : 0.99, war_pit_ra9: null },
      { season: 2026, war_bat: last.bat, war_pit: last.pit, war_pit_ra9: last.pit === null ? null : last.pit - 0.3 },
    ],
  };
}

function runs(batting: number, baserunning: number, positional: number, league: number, replacement: number): WarBatRuns {
  return {
    runs_batting: batting,
    runs_baserunning: baserunning,
    runs_fielding: 0,
    runs_positional: positional,
    runs_league: league,
    runs_replacement: replacement,
  };
}

/**
 * 「WAR은 어떻게 계산되나」의 예로 쓸 선수: 투타 겸업 + 2026 구성 항목이 있는 선수 중 예측 WAR이 가장 큰 Bravo.
 * Charlie(투타, 구성 항목 없음)·Delta(타자만)는 WAR이 더 커도 예가 아니다.
 * Bravo 2026 타자 WAR 3.21, 구성 합 24.6 − 1.2 + 0 − 13.3 + 0.9 + 17.4 = 28.4점 (선수 비율 8.85 — 화면에 적지 않는다, E29)
 */
function exampleRows(): WarRow[] {
  return [
    warRow(101, "Two Way Alpha", "both", { bat: 2.4, pit: 1.6 }, { bat: 2.31, pit: 1.2, runs: runs(20, 2, -10, 1, 12) }),
    warRow(102, "Two Way Bravo", "both", { bat: 3.95, pit: 1.6 }, { bat: 3.21, pit: 1.1, runs: runs(24.6, -1.2, -13.3, 0.9, 17.4) }),
    warRow(103, "Two Way Charlie", "both", { bat: 5.1, pit: 2.6 }, { bat: 4.4, pit: 2.2, runs: null }),
    warRow(104, "Bat Only Delta", "bat", { bat: 8.8, pit: null }, { bat: 7.7, pit: null, runs: runs(50, 3, 7, 1, 20) }),
    warRow(105, "Pitch Only Echo", "pit", { bat: null, pit: 3.3 }, { bat: null, pit: 4.1, runs: null }),
  ];
}

function handMadeWar(): WarFile {
  return {
    ...syntheticWar,
    runs_per_win: { season: 2026, value: 9.71 },
    rows: exampleRows(),
    evaluations: evaluations(),
    final_test: {
      target_season: 2026,
      ran_at: "2026-11-21T03:00:00Z",
      war_params_sha256: "d".repeat(64),
      reruns: [],
    },
  };
}

// ---------------------------------------------------------------------------
// 도우미
// ---------------------------------------------------------------------------

const PENDING = Symbol("pending");

/** validation.json은 합성 샘플(또는 주어진 것), war.json은 body(PENDING이면 끝나지 않음). 다른 주소는 404 */
function serve(war: unknown, warStatus = 200, validation: unknown = syntheticValidation) {
  const fetchMock = vi.fn(async (input: unknown): Promise<Response> => {
    const url = String(input);
    if (url === "/data/validation.json") return new Response(JSON.stringify(validation), { status: 200 });
    if (url === "/data/war.json") {
      if (war === PENDING) return new Promise<Response>(() => {});
      return new Response(JSON.stringify(war), { status: warStatus });
    }
    return new Response("not found", { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderPage() {
  return render(
    <Providers>
      <ValidationPage />
    </Providers>,
  );
}

const WAR_SEASON = "WAR 목표 시즌";

/** 기존 검증 구역(validation.json)을 받은 뒤 생기는 상세 표 선택 상자 */
async function validationLoaded() {
  await screen.findByRole("combobox", { name: "목표 시즌" }, { timeout: 3000 });
}

/** war.json을 받은 뒤 생기는 WAR 시험 표의 목표 시즌 선택 상자 (기존 "목표 시즌"과 이름이 겹치지 않게) */
async function warLoaded() {
  await validationLoaded();
  await screen.findByRole("combobox", { name: WAR_SEASON }, { timeout: 3000 });
}

const norm = (s: string | null | undefined) => (s ?? "").replace(/\s+/g, " ").trim();

/** h2 제목이 든 section — 기존 구역처럼 VStack as="section" aria-labelledby */
function sectionOf(name: string | RegExp): HTMLElement {
  const heading = screen.getByRole("heading", { level: 2, name });
  const section = heading.closest("section");
  if (section === null) throw new Error(`제목 ${String(name)}이(가) section 안에 있어야 한다`);
  return section;
}

const testSection = () => sectionOf(/^WAR 예측/);
const howSection = () => sectionOf("WAR은 어떻게 계산되나");

/** 표 밖의 글 — 표를 뺀 사본의 글자 */
function proseText(root: HTMLElement): string {
  const copy = root.cloneNode(true) as HTMLElement;
  copy.querySelectorAll("table").forEach((t) => t.remove());
  return norm(copy.textContent);
}

/** 조각을 모두 담은, 표 밖의 가장 안쪽 짧은 요소들 (한 문장·한 단락) */
function proseWith(root: HTMLElement, parts: string[]): HTMLElement[] {
  const ok = (el: Element) => parts.every((p) => (el.textContent ?? "").includes(p));
  return Array.from(root.querySelectorAll<HTMLElement>("*")).filter(
    (el) =>
      ok(el) &&
      !Array.from(el.children).some(ok) &&
      el.closest("table") === null &&
      el.querySelector("table") === null &&
      (el.textContent ?? "").length < 600,
  );
}

function expectProseWith(root: HTMLElement, parts: string[]): HTMLElement {
  const found = proseWith(root, parts);
  expect(found.length, `표 밖 한 문장에 ${parts.join(" · ")}`).toBeGreaterThan(0);
  return found[0]!;
}

/** 글자들이 이 순서로 나오는지 */
function expectInOrder(text: string, parts: string[]) {
  let at = -1;
  for (const p of parts) {
    const next = text.indexOf(p, at + 1);
    expect(next, `"${p}"가 앞 조각 뒤에 있어야 한다: ${text}`).toBeGreaterThan(at);
    at = next;
  }
}

/** WAR 시험 표 — "이 모델" 열 머리가 있는 표 */
function testTable(): HTMLElement {
  const table = within(testSection())
    .queryAllByRole("table")
    .find((t) => within(t).queryAllByRole("columnheader").some((h) => norm(h.textContent) === "이 모델"));
  if (table === undefined) throw new Error("WAR 시험 표(이 모델 열)가 없습니다");
  return table;
}

function headers(table: HTMLElement): string[] {
  return within(table)
    .getAllByRole("columnheader")
    .map((h) => norm(h.textContent));
}

/** 머리 행을 뺀 본문 행 */
function dataRows(table: HTMLElement): HTMLElement[] {
  return within(table)
    .getAllByRole("row")
    .filter((r) => within(r).queryAllByRole("columnheader").length === 0);
}

/** 열 머리 순서로 그 행의 칸 글자 */
function cellText(table: HTMLElement, row: HTMLElement, header: string): string {
  const idx = headers(table).indexOf(header);
  expect(idx, `열 머리 ${header}`).toBeGreaterThanOrEqual(0);
  const cell = row.children[idx];
  expect(cell, `${header} 칸`).toBeDefined();
  return norm(cell!.textContent);
}

const RESULT = {
  win: "두 기준선보다 낮음",
  losePrev: "작년 그대로에 짐",
  loseLeague: "리그 평균 비율에 짐",
  loseBoth: "두 기준선에 짐",
} as const;

/** 시즌 표의 여섯 줄이 CELLS 순서·값·결과 글 그대로인지 */
function expectSeasonTable(season: number, results: string[]) {
  const table = testTable();
  const rows = dataRows(table);
  expect(rows, `${season} 시험 표는 여섯 줄`).toHaveLength(6);
  SEASON_VALUES[season]!.forEach(([n, mae, prev, league], i) => {
    const row = rows[i]!;
    const cell = CELLS[i]!;
    const label = `${season} ${i + 1}번째 줄`;
    expect(cellText(table, row, "무엇을"), label).toBe(cell.what);
    expect(cellText(table, row, "누구에게"), label).toBe(`${cell.who} ${fmtCount(n)}명`);
    expect(cellText(table, row, "이 모델"), label).toBe(fmtWarMae(mae));
    expect(cellText(table, row, "작년 그대로"), label).toBe(fmtWarMae(prev));
    expect(cellText(table, row, "리그 평균 비율"), label).toBe(fmtWarMae(league));
    const result = cellText(table, row, "결과");
    expect(result, label).toContain(results[i]!);
    // 이긴 칸에는 "짐"이 없고, 진 칸에는 "낮음"이 없다 — 색만이 아니라 글로 (N10)
    if (results[i] === RESULT.win) expect(result, label).not.toContain("짐");
    else expect(result, label).not.toContain("낮음");
  });
}

/** Astryx Selector(콤보박스)나 기본 select에서 고른다 */
async function choose(label: string, option: string) {
  const control = screen.getByRole("combobox", { name: label });
  if (control instanceof HTMLSelectElement) {
    const opt = within(control).getByRole("option", { name: option }) as HTMLOptionElement;
    fireEvent.change(control, { target: { value: opt.value } });
    return;
  }
  fireEvent.click(control);
  fireEvent.click(await screen.findByRole("option", { name: option }));
}

/** 선택 상자의 선택지 글자 */
async function optionsOf(label: string): Promise<string[]> {
  const control = screen.getByRole("combobox", { name: label });
  if (control instanceof HTMLSelectElement) {
    return Array.from(control.options).map((o) => o.textContent ?? "");
  }
  fireEvent.click(control);
  const options = await screen.findAllByRole("option");
  const texts = options.map((o) => (o.textContent ?? "").trim());
  // 지금 고른 값을 다시 눌러 닫는다 — 열린 목록이 다음 질의에 섞이지 않게
  const selected = options.find((o) => o.getAttribute("aria-selected") === "true") ?? options[0]!;
  fireEvent.click(selected);
  await waitFor(() => expect(screen.queryAllByRole("option")).toHaveLength(0));
  return texts;
}

function expectHeadingOrder() {
  const levels = screen
    .getAllByRole("heading")
    .map((h) => Number(h.getAttribute("aria-level") ?? h.tagName.slice(1)));
  expect(levels[0]).toBe(1);
  expect(levels.filter((l) => l === 1)).toHaveLength(1);
  levels.forEach((l, i) => {
    if (i > 0) expect(l, `제목 단계가 건너뛴다: ${levels.join(",")}`).toBeLessThanOrEqual(levels[i - 1]! + 1);
  });
}

const follows = (a: Node, b: Node) => (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0;

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

// ---------------------------------------------------------------------------

describe("검증 화면 WAR — 불러오기·자리 (SA-8)", () => {
  it("validation.json과 war.json을 한 번씩만 받는다 — 선수·예측표 파일은 받지 않는다", async () => {
    const fetchMock = serve(handMadeWar());
    renderPage();
    await warLoaded();
    const calls = fetchMock.mock.calls.map((c) => String(c[0])).sort();
    expect(calls).toEqual(["/data/validation.json", "/data/war.json"]);
  });

  it("WAR 구역은 기존 여섯 지표 구역(최종 검증 · ± 범위 · 모델별 표) 아래 — WAR 예측 → WAR은 어떻게 계산되나", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    const h2 = (name: string | RegExp) => screen.getByRole("heading", { level: 2, name });
    const order = [h2(/최종 검증/), h2(/± 범위/), h2(/모델별·지표별 오차/), h2(/^WAR 예측/), h2("WAR은 어떻게 계산되나")];
    order.forEach((el, i) => {
      if (i > 0) expect(follows(order[i - 1]!, el), `${order[i - 1]!.textContent} → ${el.textContent}`).toBe(true);
    });
    // WAR 시험 표는 기존 모델별 표보다 뒤에 있다
    const detail = screen.getByRole("combobox", { name: "목표 시즌" });
    expect(follows(detail, screen.getByRole("combobox", { name: WAR_SEASON }))).toBe(true);
    expectHeadingOrder();
  });

  it("war.json을 받는 동안 기존 구역은 그대로 두고 WAR 구역만 불러오는 중", async () => {
    serve(PENDING);
    renderPage();
    await validationLoaded();
    expect(screen.getByRole("heading", { level: 2, name: /모델별·지표별 오차/ })).toBeInTheDocument();
    const visible = screen.queryAllByText(/WAR.*불러오는 중/);
    const labelled = screen.queryAllByLabelText(/WAR.*불러오는 중/);
    expect(visible.length + labelled.length, "WAR 로딩 표시").toBeGreaterThan(0);
    expect(screen.queryByRole("combobox", { name: WAR_SEASON })).toBeNull();
  });

  it("war.json을 받지 못하면(500) WAR 오류만 보여 주고 기존 구역은 그대로", async () => {
    serve({}, 500);
    renderPage();
    await validationLoaded();
    expect(await screen.findByText(/WAR.*불러오지 못했/, undefined, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: /모델별·지표별 오차/ })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "목표 시즌" })).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: WAR_SEASON })).toBeNull();
  });

  it("모양이 틀린 war.json(출처가 MLB Stats API가 아님)도 오류 — loadWar로 받는다", async () => {
    serve({ ...handMadeWar(), source: "FanGraphs" });
    renderPage();
    await validationLoaded();
    expect(await screen.findByText(/WAR.*불러오지 못했/, undefined, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: WAR_SEASON })).toBeNull();
    expect(screen.queryAllByText(RESULT.win)).toHaveLength(0);
  });
});

describe("검증 화면 WAR — 2026 시험 표 (SA-8, D36)", () => {
  it("제목 「WAR 예측 — 2026 시험」, 고정 설정 표시(다시 돌림 0번), 안내 문구", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    expect(screen.getByRole("heading", { level: 2, name: "WAR 예측 — 2026 시험" })).toBeInTheDocument();
    const section = testSection();
    expect(within(section).getAllByText(/고정한 설정으로 한 번 · 다시 돌림 0번/).length).toBeGreaterThan(0);
    expect(proseText(section)).toContain(WAR_TEST_INTRO);
  });

  it("다시 돌림 횟수는 final_test.reruns 길이", async () => {
    const war = handMadeWar();
    war.final_test = {
      ...war.final_test!,
      reruns: [
        { ran_at: "2026-11-22T01:00:00Z", reason: "합성 사유 1" },
        { ran_at: "2026-11-23T01:00:00Z", reason: "합성 사유 2" },
      ],
    };
    serve(war);
    renderPage();
    await warLoaded();
    const section = testSection();
    expect(within(section).getAllByText(/고정한 설정으로 한 번 · 다시 돌림 2번/).length).toBeGreaterThan(0);
    expect(within(section).queryAllByText(/다시 돌림 0번/)).toHaveLength(0);
  });

  it("열 6개: 무엇을 · 누구에게 · 이 모델 · 작년 그대로 · 리그 평균 비율 · 결과", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    expect(headers(testTable())).toEqual(["무엇을", "누구에게", "이 모델", "작년 그대로", "리그 평균 비율", "결과"]);
  });

  it("기본 2026 — 여섯 줄이 정해진 순서로, 값은 fmtWarMae · 대상 수는 fmtCount, 결과는 글로", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    expectSeasonTable(2026, [RESULT.win, RESULT.win, RESULT.win, RESULT.win, RESULT.win, RESULT.losePrev]);
    // 경력 그룹 행(1~2시즌 등)은 시험 표에 넣지 않는다
    expect(testTable().textContent).not.toContain(fmtWarMae(GROUP_ONLY_MAE));
    expect(testTable().textContent).not.toContain("555명");
  });

  it("결과 문장: 여섯 칸 중 다섯 칸, 진 칸을 숫자와 함께 그대로, 풀타임 환산을 같이 두는 이유 — 이긴 칸 숫자는 적지 않는다", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    const section = testSection();
    expectProseWith(section, ["여섯 칸 중 다섯 칸에서 두 기준선보다 오차가 작았습니다"]);
    // 진 칸(투수 예측 WAR, 실제 WAR 2 이상): 이 모델 1.77 대 작년 그대로 1.51 — 숨기지 않는다 (D36)
    const lost = expectProseWith(section, [
      "투수 예측 WAR",
      "실제 WAR 2 이상",
      fmtWarMae(1.7654),
      "작년 그대로",
      fmtWarMae(1.5123),
    ]);
    expectInOrder(norm(lost.textContent), [fmtWarMae(1.7654), "작년 그대로", fmtWarMae(1.5123)]);
    expectProseWith(section, ["그래서 순위표에 풀타임 환산 WAR을 같이 둡니다"]);
    // 실제 WAR 2 이상 칸에서 졌으니 출전량 보수성 해석을 붙인다 (SA-8)
    expectProseWith(section, ["출전량을 보수적으로 잡는 약점"]);
    // 이긴 다섯 칸의 숫자는 진 칸으로 적지 않는다
    const prose = proseText(section);
    SEASON_VALUES[2026]!.slice(0, 5).forEach(([, mae, prev, league], i) => {
      for (const v of [mae, prev, league]) expect(prose, `${i + 1}번째 칸 ${fmtWarMae(v)}`).not.toContain(fmtWarMae(v));
    });
  });

  it("목표 시즌 선택: 2026(기본) · 2025 · 2024 — 기존 모델별 표의 '목표 시즌'과 따로", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    expect(await optionsOf(WAR_SEASON)).toEqual(["2026", "2025", "2024"]);
    // 기존 상세 표의 목표 시즌(합성 validation.json은 2025만)은 그대로
    expect(await optionsOf("목표 시즌")).toEqual(["2025"]);
  });

  it("2025로 바꾸면 2025 값과 결과 문장(여섯 칸 중 네 칸, 진 칸 둘)", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    await choose(WAR_SEASON, "2025");
    expectSeasonTable(2025, [RESULT.win, RESULT.win, RESULT.win, RESULT.win, RESULT.losePrev, RESULT.losePrev]);
    // 칸 글자를 이어 붙인 textContent는 이웃 칸(1.43 + 2.26 → "1.432.26")이 우연히 ".432"를 만들어 칸마다 본다
    const table2025 = testTable();
    expect(dataRows(table2025).map((r) => cellText(table2025, r, "이 모델"))).not.toContain(fmtWarMae(0.4321));
    const section = testSection();
    expectProseWith(section, ["여섯 칸 중 네 칸에서 두 기준선보다 오차가 작았습니다"]);
    expect(proseText(section)).not.toContain("여섯 칸 중 다섯 칸");
    expectProseWith(section, ["타자 예측 WAR", "실제 WAR 2 이상", fmtWarMae(1.6611), "작년 그대로", fmtWarMae(1.6422)]);
    expectProseWith(section, ["투수 예측 WAR", "실제 WAR 2 이상", fmtWarMae(1.6211), "작년 그대로", fmtWarMae(1.4311)]);
  });

  it("2024: '리그 평균 비율에 짐' · '두 기준선에 짐'도 글로 적고, 진 칸에 진 기준선 숫자를 적는다", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    await choose(WAR_SEASON, "2024");
    expectSeasonTable(2024, [RESULT.win, RESULT.win, RESULT.loseLeague, RESULT.win, RESULT.win, RESULT.loseBoth]);
    const section = testSection();
    expectProseWith(section, ["여섯 칸 중 네 칸에서 두 기준선보다 오차가 작았습니다"]);
    expectProseWith(section, [`타자 ${FULL_TIME_SHORT}`, "주전급", fmtWarMae(1.5511), "리그 평균 비율", fmtWarMae(1.5011)]);
    expectProseWith(section, [
      "투수 예측 WAR",
      "실제 WAR 2 이상",
      fmtWarMae(1.6611),
      "작년 그대로",
      fmtWarMae(1.4911),
      "리그 평균 비율",
      fmtWarMae(1.5733),
    ]);
  });

  it("여섯 칸 모두 이기면 진 칸을 적지 않는다", async () => {
    const war = handMadeWar();
    war.evaluations = war.evaluations.map((r) =>
      r.target_season === 2026 && r.role === "pit" && r.metric === "war" && r.group === "war2plus" ? { ...r, mae: 1.4012 } : r,
    );
    serve(war);
    renderPage();
    await warLoaded();
    const table = testTable();
    for (const row of dataRows(table)) expect(cellText(table, row, "결과")).toContain(RESULT.win);
    const prose = proseText(testSection());
    expect(prose).toContain("여섯 칸");
    expect(prose).toContain("두 기준선보다 오차가 작았습니다");
    expect(prose).not.toContain("여섯 칸 중 다섯 칸");
    expect(prose).not.toContain("에 짐");
    expect(prose).not.toContain(fmtWarMae(1.5123));
  });

  // Refactor 리뷰에서 더함 — 값이 빠진 칸 · 다시 돌린 사유 · 모두 진 경우
  it("이 모델 오차가 없는(null) 칸은 '두 기준선보다 낮음'으로 적지 않고, 이긴 칸 수에도 넣지 않는다", async () => {
    const war = handMadeWar();
    war.evaluations = war.evaluations.map((r) =>
      r.target_season === 2026 && r.role === "bat" && r.metric === "war" && r.group === "all" ? { ...r, mae: null } : r,
    );
    serve(war);
    renderPage();
    await warLoaded();
    const table = testTable();
    const first = dataRows(table)[0]!;
    expect(cellText(table, first, "이 모델")).toBe("—");
    expect(cellText(table, first, "결과")).not.toContain(RESULT.win);
    expect(cellText(table, first, "결과")).not.toContain("짐");
    const prose = proseText(testSection());
    expect(prose).toContain("다섯 칸 중 네 칸에서 두 기준선보다 오차가 작았습니다");
    expect(prose).not.toContain("여섯 칸");
  });

  it("다시 돌렸으면 기록된 사유를 숨기지 않고 적는다 (D36)", async () => {
    const war = handMadeWar();
    war.final_test = { ...war.final_test!, reruns: [{ ran_at: "2026-11-22T01:00:00Z", reason: "합성 사유 하나" }] };
    serve(war);
    renderPage();
    await warLoaded();
    expect(proseText(testSection())).toContain("기록된 사유: 합성 사유 하나");
  });

  it("여섯 칸 모두 지면 '모두 작지 않았다'가 아니라 이긴 칸이 없다고 적고, 진 칸을 다 적는다", async () => {
    const war = handMadeWar();
    war.evaluations = war.evaluations.map((r) => (r.target_season === 2026 ? { ...r, mae: 9.5 } : r));
    serve(war);
    renderPage();
    await warLoaded();
    const prose = proseText(testSection());
    expect(prose).toContain("여섯 칸 중 두 기준선보다 오차가 작은 칸은 없었습니다");
    expect(prose).not.toContain("모두 두 기준선보다 오차가 작았습니다");
    expect(prose).toContain("그래서 순위표에 풀타임 환산 WAR을 같이 둡니다");
    const table = testTable();
    for (const row of dataRows(table)) expect(cellText(table, row, "결과")).toContain(RESULT.loseBoth);
  });
});

describe("검증 화면 WAR — 2026 시험 전 (final_test null)", () => {
  it("합성 샘플: 2026(기본)은 '아직 시험하지 않았습니다', 2025 · 2024는 과거 검증 값", async () => {
    serve(syntheticWar);
    renderPage();
    await warLoaded();
    const section = testSection();
    expect(norm(section.textContent)).toContain("아직 시험하지 않았습니다");
    expect(within(section).queryAllByText(/다시 돌림/)).toHaveLength(0);
    expect(within(section).queryAllByText(RESULT.win)).toHaveLength(0);
    expect(proseText(section)).not.toContain("여섯 칸 중");
    expect(await optionsOf(WAR_SEASON)).toEqual(["2026", "2025", "2024"]);

    // 2025: 투수 풀타임 환산(주전급 68명) .930 · 1.13 · 1.31
    await choose(WAR_SEASON, "2025");
    let table = testTable();
    let row = dataRows(table).find((r) => cellText(table, r, "무엇을") === `투수 ${FULL_TIME_SHORT}`);
    expect(row, "2025 투수 풀타임 환산 줄").toBeDefined();
    expect(cellText(table, row!, "누구에게")).toBe(`주전급 ${fmtCount(68)}명`);
    expect(cellText(table, row!, "이 모델")).toBe(fmtWarMae(0.93));
    expect(cellText(table, row!, "작년 그대로")).toBe(fmtWarMae(1.128));
    expect(cellText(table, row!, "리그 평균 비율")).toBe(fmtWarMae(1.311));
    expect(cellText(table, row!, "결과")).toContain(RESULT.win);
    expect(norm(testSection().textContent)).not.toContain("아직 시험하지 않았습니다");

    // 2024: 타자 예측 WAR(전원 1,118명) .500 · .591 · .575
    await choose(WAR_SEASON, "2024");
    table = testTable();
    row = dataRows(table).find((r) => cellText(table, r, "누구에게") === `전원 ${fmtCount(1118)}명`);
    expect(row, "2024 타자 예측 WAR 전원 줄").toBeDefined();
    expect(cellText(table, row!, "무엇을")).toBe("타자 예측 WAR");
    expect(cellText(table, row!, "이 모델")).toBe(fmtWarMae(0.5));
    expect(cellText(table, row!, "작년 그대로")).toBe(fmtWarMae(0.591));
    expect(cellText(table, row!, "리그 평균 비율")).toBe(fmtWarMae(0.575));
  });

  it("final_test가 null이면 evaluations에 2026 행이 있어도 2026 값을 보여 주지 않는다", async () => {
    serve({ ...handMadeWar(), final_test: null });
    renderPage();
    await warLoaded();
    const section = testSection();
    const text = norm(section.textContent);
    expect(text).toContain("아직 시험하지 않았습니다");
    for (const [, mae] of SEASON_VALUES[2026]!) expect(text).not.toContain(fmtWarMae(mae));
    await choose(WAR_SEASON, "2025");
    expectSeasonTable(2025, [RESULT.win, RESULT.win, RESULT.win, RESULT.win, RESULT.losePrev, RESULT.losePrev]);
  });
});

describe("검증 화면 WAR — WAR은 어떻게 계산되나 (SA-8, D40 · D41 · E29)", () => {
  it("WAR_WHAT 문단, 두 칸 제목(타자 WAR · 투수 WAR — FIP 기반)", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    const section = howSection();
    expect(norm(section.textContent)).toContain(WAR_WHAT);
    expect(within(section).getByRole("heading", { level: 3, name: "타자 WAR" })).toBeInTheDocument();
    expect(within(section).getByRole("heading", { level: 3, name: PIT_WAR_TITLE })).toBeInTheDocument();
  });

  it("타자 WAR: 구성 항목 이름 6개가 순서대로, 그 뒤에 BAT_WAR_SUMMARY, 그다음 투수 칸", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    expectInOrder(norm(howSection().textContent), [
      WAR_WHAT,
      "타자 WAR",
      ...BAT_RUNS_PARTS.map((p) => p.name),
      BAT_WAR_SUMMARY,
      PIT_WAR_TITLE,
    ]);
  });

  it("예: 투타 겸업 + 구성 항목이 있는 선수 중 예측 WAR이 가장 큰 선수 — 2026 타자 WAR(fmtWar2) = 구성 합(점)(타격 · 포지션, fmtRuns)", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    const section = howSection();
    const example = expectProseWith(section, ["Two Way Bravo", `2026 타자 WAR ${fmtWar2(3.21)}`, "28.4점"]);
    const text = norm(example.textContent);
    expectInOrder(text, ["Two Way Bravo", `2026 타자 WAR ${fmtWar2(3.21)}`, "28.4점", `타격 ${fmtRuns(24.6)}`]);
    expect(text).toMatch(new RegExp(`포지션( 보정)? ${escapeRe(fmtRuns(-13.3))}`));
    // 2027 예측(3.95)이 아니라 2026 실제 타자 WAR을 쓴다
    expect(text).not.toContain(fmtWar2(3.95));
    // 선수별 점 → 승 비율(28.4 ÷ 3.21 ≈ 8.85)은 적지 않는다 (E29)
    expect(norm(section.textContent)).not.toMatch(/8\.8[0-9]?/);
    // 예가 아닌 선수: 구성 항목 없는 투타(Charlie) · 타자만(Delta) · WAR이 작은 투타(Alpha) · 투수만(Echo)
    for (const other of ["Two Way Alpha", "Two Way Charlie", "Bat Only Delta", "Pitch Only Echo"]) {
      expect(norm(section.textContent), other).not.toContain(other);
    }
  });

  it("합성 샘플의 예: Sample Batter 2026 타자 WAR 3.69 = 36.2점(타격 +29.2, 포지션 −14.5 …)", async () => {
    serve(syntheticWar);
    renderPage();
    await warLoaded();
    const example = expectProseWith(howSection(), ["Sample Batter", `2026 타자 WAR ${fmtWar2(3.69)}`, "36.2점"]);
    const text = norm(example.textContent);
    expectInOrder(text, ["Sample Batter", `2026 타자 WAR ${fmtWar2(3.69)}`, "36.2점", `타격 ${fmtRuns(29.2)}`]);
    expect(text).toMatch(new RegExp(`포지션( 보정)? ${escapeRe(fmtRuns(-14.5))}`));
    expect(norm(howSection().textContent)).not.toContain("Sample Pitcher");
  });

  it("예로 쓸 선수(투타 겸업 + 구성 항목)가 없으면 예를 뺀다 — 나머지 설명은 그대로", async () => {
    const war = handMadeWar();
    war.rows = war.rows.filter((r) => !(r.roles === "both" && r.prev_bat_runs !== null));
    serve(war);
    renderPage();
    await warLoaded();
    const text = norm(howSection().textContent);
    expect(text).not.toContain("2026 타자 WAR");
    expect(text).not.toMatch(/\d점\(/);
    for (const name of ["Two Way Charlie", "Bat Only Delta", "Pitch Only Echo"]) expect(text, name).not.toContain(name);
    expect(text).toContain(BAT_WAR_SUMMARY);
    expect(text).toContain(PIT_WAR_STABILITY.note);
  });

  it("투수 WAR: 정의 3줄 × 작년 → 올해 상관(2024 · 2025) 표와 설명", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    const section = howSection();
    expect(norm(section.textContent)).toContain(PIT_WAR_STABILITY.caption);
    const table = within(section)
      .queryAllByRole("table")
      .find((t) => (t.textContent ?? "").includes(PIT_WAR_STABILITY.rows[0].name));
    expect(table, "투수 WAR 정의 표").toBeDefined();
    const rows = dataRows(table!);
    expect(rows).toHaveLength(PIT_WAR_STABILITY.rows.length);
    PIT_WAR_STABILITY.rows.forEach((def, i) => {
      expectInOrder(norm(rows[i]!.textContent), [def.name, fmtWar2(def.corr[0]), fmtWar2(def.corr[1])]);
    });
    expect(norm(section.textContent)).toContain(PIT_WAR_STABILITY.note);
  });

  it("카드 셋 WAR_FACTS — 제목·본문이 순서대로, 투수 칸 뒤에", async () => {
    serve(handMadeWar());
    renderPage();
    await warLoaded();
    expectInOrder(
      norm(howSection().textContent),
      [PIT_WAR_STABILITY.note, ...WAR_FACTS.flatMap((f) => [f.title, f.body])],
    );
  });
});
