// 검증 화면 (SA-4, SPEC F15·D29·D31) — 레인 C Red 테스트
// fetch는 vi.stubGlobal로 합성 샘플·손으로 만든 보고서를 준다. 네트워크는 쓰지 않는다.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ValidationPage from "@/app/validation/page";
import { Providers } from "@/app/providers";
import type {
  BandRow,
  EvalRow,
  GroupName,
  Metric,
  ModelName,
  Role,
  ValidationReport,
} from "@/src/data/types";
import { fmtBand, fmtCount, fmtMae } from "@/src/format";
import { DISCLAIMER_SHORT } from "@/src/site";

const ROOT = join(import.meta.dirname, "..", "..");
const synthetic = JSON.parse(
  readFileSync(join(ROOT, "tests/fixtures/synthetic/web/validation.json"), "utf-8"),
) as ValidationReport;

// ---------------------------------------------------------------------------
// 손으로 만든 보고서 — 실제 값과 다른 숫자로, 화면이 값을 데이터에서 읽는지 본다
// ---------------------------------------------------------------------------

function row(
  target_season: number,
  role: Role,
  model: ModelName,
  metric: Metric,
  group: GroupName,
  n: number,
  mae: number,
  league: number | null,
  prev: number | null,
): EvalRow {
  return {
    target_season,
    role,
    model,
    metric,
    group,
    n,
    mae,
    mae_weighted: null,
    baseline_mae_league: league,
    baseline_mae_prev: prev,
    excluded: 0,
  };
}

/** 한 그룹의 모델 네 개(확장·가중·직전 시즌·리그 평균) 행. 리그 평균 모델은 비율 지표만 */
function modelRows(
  season: number,
  role: Role,
  metric: Metric,
  group: GroupName,
  n: number,
  v: { ext?: number; wtd: number; league: number | null; prev: number },
): EvalRow[] {
  const out: EvalRow[] = [];
  if (v.ext !== undefined) out.push(row(season, role, "extended", metric, group, n, v.ext, v.league, v.prev));
  out.push(row(season, role, "weighted", metric, group, n, v.wtd, v.league, v.prev));
  out.push(row(season, role, "baseline", metric, group, n, v.prev, v.league, v.prev));
  if (v.league !== null) out.push(row(season, role, "league", metric, group, n, v.league, v.league, v.prev));
  return out;
}

const REASON = "합성 사유: 소표본 특성을 리그 평균 쪽으로 당기도록 고친 뒤 다시 돌렸다";

function band(role: Role, metric: BandRow["metric"], group: BandRow["group"], n: number, q80: number): BandRow {
  return { role, metric, group, n, q80, targets: [2024, 2025, 2026] };
}

function handMadeReport(): ValidationReport {
  const rows: EvalRow[] = [
    // 2026 전체 — 여섯 지표 모두 두 기준선보다 낮다
    ...modelRows(2026, "bat", "ops", "all", 300, { ext: 0.0655, wtd: 0.0671, league: 0.0744, prev: 0.0912 }),
    ...modelRows(2026, "bat", "pa", "all", 1000, { ext: 70.04, wtd: 70.04, league: null, prev: 73.31 }),
    ...modelRows(2026, "bat", "hr", "all", 1000, { ext: 2.5012, wtd: 2.553, league: null, prev: 2.8877 }),
    ...modelRows(2026, "pit", "era", "all", 280, { ext: 0.8811, wtd: 0.902, league: 0.9562, prev: 1.3012 }),
    ...modelRows(2026, "pit", "k_bb_pct", "all", 280, { ext: 3.7123, wtd: 3.8001, league: 4.5012, prev: 4.6123 }),
    ...modelRows(2026, "pit", "ip", "all", 1400, { ext: 16.92, wtd: 16.92, league: null, prev: 17.71 }),
    // 2026 그룹 — 1~2시즌 HR(1.81 대 1.70), 3~5시즌 IP(19.9 대 19.1)만 직전 시즌에 진다
    ...modelRows(2026, "bat", "ops", "seasons_1_2", 90, { ext: 0.0702, wtd: 0.0711, league: 0.079, prev: 0.1011 }),
    ...modelRows(2026, "bat", "pa", "seasons_1_2", 400, { ext: 45.12, wtd: 45.12, league: null, prev: 47.03 }),
    ...modelRows(2026, "bat", "hr", "seasons_1_2", 400, { ext: 1.8123, wtd: 1.8301, league: null, prev: 1.7004 }),
    ...modelRows(2026, "bat", "ops", "seasons_3_5", 210, { ext: 0.0633, wtd: 0.0649, league: 0.0721, prev: 0.0899 }),
    ...modelRows(2026, "bat", "pa", "seasons_3_5", 600, { ext: 80.11, wtd: 80.11, league: null, prev: 82.4 }),
    ...modelRows(2026, "bat", "hr", "seasons_3_5", 600, { ext: 2.9012, wtd: 2.95, league: null, prev: 3.1012 }),
    ...modelRows(2026, "pit", "era", "seasons_1_2", 80, { ext: 0.9512, wtd: 0.97, league: 1.0012, prev: 1.4012 }),
    ...modelRows(2026, "pit", "k_bb_pct", "seasons_1_2", 80, { ext: 3.9012, wtd: 3.95, league: 4.6012, prev: 4.8012 }),
    ...modelRows(2026, "pit", "ip", "seasons_1_2", 500, { ext: 12.11, wtd: 12.11, league: null, prev: 12.94 }),
    ...modelRows(2026, "pit", "era", "seasons_3_5", 200, { ext: 0.8512, wtd: 0.87, league: 0.9312, prev: 1.2512 }),
    ...modelRows(2026, "pit", "k_bb_pct", "seasons_3_5", 200, { ext: 3.6012, wtd: 3.65, league: 4.4012, prev: 4.5012 }),
    ...modelRows(2026, "pit", "ip", "seasons_3_5", 900, { ext: 19.87, wtd: 19.87, league: null, prev: 19.12 }),
    ...modelRows(2026, "bat", "ops", "prev_played", 280, { ext: 0.0644, wtd: 0.0662, league: 0.0733, prev: 0.0901 }),
    // 2025·2024 전체
    ...modelRows(2025, "bat", "ops", "all", 310, { ext: 0.0688, wtd: 0.0701, league: 0.0766, prev: 0.0855 }),
    ...modelRows(2025, "pit", "era", "all", 290, { ext: 0.9001, wtd: 0.9101, league: 0.9612, prev: 1.2712 }),
    ...modelRows(2024, "bat", "ops", "all", 305, { ext: 0.071, wtd: 0.0722, league: 0.0788, prev: 0.0977 }),
  ];
  return {
    meta: synthetic.meta,
    final_test: {
      target_season: 2026,
      ran_at: "2026-11-20T01:00:00Z",
      params_sha256: "b".repeat(64),
      reruns: [{ ran_at: "2026-11-20T02:00:00Z", reason: REASON }],
      first_run: {
        ran_at: "2026-11-20T01:00:00Z",
        mae: { ops: 0.0701, k_bb_pct: 3.911, era: 0.9123, hr: 2.4449, pa: 70.04, ip: 16.92 },
      },
    },
    rows,
    bands: [
      band("bat", "ops", "seasons_1_2", 250, 0.111),
      band("bat", "ops", "seasons_3_5", 700, 0.099),
      band("pit", "era", "seasons_1_2", 260, 1.52),
      band("pit", "era", "seasons_3_5", 610, 1.41),
      band("pit", "k_bb_pct", "seasons_1_2", 260, 6.1),
      band("pit", "k_bb_pct", "seasons_3_5", 610, 5.7),
    ],
  };
}

// ---------------------------------------------------------------------------
// 도우미
// ---------------------------------------------------------------------------

/** /data/validation.json에 body를 주는 가짜 fetch. 다른 주소는 404 */
function serve(body: unknown, status = 200) {
  const fetchMock = vi.fn(async (input: unknown) =>
    String(input) === "/data/validation.json"
      ? new Response(JSON.stringify(body), { status })
      : new Response("not found", { status: 404 }),
  );
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

/** 데이터를 받은 뒤에만 생기는 상세 표 선택 상자를 기다린다 */
async function loaded() {
  await screen.findByRole("combobox", { name: "목표 시즌" }, { timeout: 3000 });
}

/** 조각을 모두 담고(제외 조각은 없이) 자식 중엔 그런 요소가 없는 가장 안쪽 요소들 */
function innermostWith(parts: string[], exclude: string[] = []): HTMLElement[] {
  const ok = (el: Element) => {
    const text = el.textContent ?? "";
    return parts.every((p) => text.includes(p)) && exclude.every((x) => !text.includes(x));
  };
  return Array.from(document.body.querySelectorAll<HTMLElement>("*")).filter(
    (el) => ok(el) && !Array.from(el.children).some(ok),
  );
}

/** 문장 하나(표·카드 묶음이 아닌 짧은 요소)가 조각을 모두 담는지 */
function expectSentenceWith(parts: string[], exclude: string[] = []) {
  const found = innermostWith(parts, exclude).filter(
    (el) => el.querySelector("table") === null && (el.textContent ?? "").length < 600,
  );
  expect(found.length, `한 문장에 ${parts.join(" · ")}`).toBeGreaterThan(0);
}

const count = (text: string | null, part: string) => (text ?? "").split(part).length - 1;

/** 2026 최종 검증 카드 — 제목(지표 이름)에서 올라가 "이 모델"이 한 번만 든 가장 작은 요소 */
function cardFor(metricTitle: string): HTMLElement {
  for (const title of screen.getAllByText(metricTitle, { exact: true })) {
    let node: HTMLElement | null = title;
    while (node !== null && count(node.textContent, "이 모델") === 0) node = node.parentElement;
    if (node !== null && count(node.textContent, "이 모델") === 1) return node;
  }
  throw new Error(`카드를 찾지 못했습니다: ${metricTitle}`);
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

/** 헤더 칸으로 표를 찾는다 */
function tableWithHeader(header: RegExp): HTMLElement {
  const table = screen
    .getAllByRole("table")
    .find((t) => within(t).queryAllByRole("columnheader").some((h) => header.test(h.textContent ?? "")));
  if (table === undefined) throw new Error(`헤더 ${header}인 표가 없습니다`);
  return table;
}

/** 표 본문 행 중 조각을 모두 담은 행 */
function rowsWith(table: HTMLElement, parts: string[]): HTMLElement[] {
  return within(table)
    .getAllByRole("row")
    .filter((r) => parts.every((p) => (r.textContent ?? "").includes(p)));
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

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

// ---------------------------------------------------------------------------

describe("검증 화면 — 공통·상태", () => {
  it("헤더(검증 선택)·바닥글, 제목·리드, validation.json 한 번만 받는다", async () => {
    const fetchMock = serve(handMadeReport());
    renderPage();
    await loaded();
    expect(screen.getByRole("link", { current: "page" })).toHaveTextContent("검증");
    expect(screen.getByText(DISCLAIMER_SHORT)).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("예측은 얼마나 맞았나");
    // 리드 (SA-4 문구): 2024·2025로 설정, 2026은 고정해 시험
    expectSentenceWith(["2024·2025 시즌만으로 정했", "2026 시즌을 시험"]);
    expect(fetchMock.mock.calls.map((c) => String(c[0]))).toEqual(["/data/validation.json"]);
  });

  it("받는 동안 로딩 상태를 보여 준다", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => new Promise<Response>(() => {})),
    );
    renderPage();
    const visible = screen.queryAllByText(/불러오는 중/);
    const labelled = screen.queryAllByLabelText(/불러오는 중/);
    expect(visible.length + labelled.length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { current: "page" })).toHaveTextContent("검증");
  });

  it("받지 못하면 오류를 보여 준다", async () => {
    serve({}, 500);
    renderPage();
    expect(await screen.findByText(/불러오지 못했/, undefined, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "목표 시즌" })).toBeNull();
    expect(screen.getByText(DISCLAIMER_SHORT)).toBeInTheDocument();
  });

  it("파일이 없으면(404) 아직 없다는 안내를 따로 보여 준다", async () => {
    serve({}, 404);
    renderPage();
    expect(await screen.findByText(/불러오지 못했/, undefined, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByText(/검증 자료 파일이 아직 없습니다/)).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "목표 시즌" })).toBeNull();
  });

  it("행·범위가 비어 있어도 무너지지 않고 빈 상태를 보여 준다", async () => {
    serve({ ...synthetic, final_test: null, rows: [], bands: [] });
    renderPage();
    await loaded();
    expect(screen.getByText("± 범위 자료가 없습니다")).toBeInTheDocument();
    expect(screen.getByText("이 조합의 검증 결과가 없습니다")).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: /최종 검증/ })).toBeInTheDocument();
  });

  it("최종 검증은 있는데 그 시즌 행이 없으면 카드 대신 빈 상태", async () => {
    const report = handMadeReport();
    report.rows = report.rows.filter((r) => r.target_season !== 2026);
    serve(report);
    renderPage();
    await loaded();
    expect(screen.queryAllByText("이 모델")).toHaveLength(0);
    expect(screen.getByText("2026 최종 검증 행이 자료에 없습니다")).toBeInTheDocument();
    expect(screen.queryAllByText(/여섯 지표 모두/)).toHaveLength(0);
  });

  it("제목 단계가 차례대로다 (h1 하나, 건너뛰지 않음) · 섹션 제목", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();
    expectHeadingOrder();
    expect(screen.getByRole("heading", { level: 2, name: /2026 최종 검증/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: /± 범위/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: /모델별·지표별 오차/ })).toBeInTheDocument();
  });
});

describe("검증 화면 — 2026 최종 검증 카드 (SA-4)", () => {
  it("지표 6개: 최종 모델(확장)·리그 평균(비율 지표만)·직전 시즌, 값은 fmtMae, 대상 수", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();

    const expected: Array<[string, Metric, number, number | null, number, number]> = [
      ["OPS", "ops", 0.0655, 0.0744, 0.0912, 300],
      ["K−BB%", "k_bb_pct", 3.7123, 4.5012, 4.6123, 280],
      ["ERA", "era", 0.8811, 0.9562, 1.3012, 280],
      ["PA", "pa", 70.04, null, 73.31, 1000],
      ["IP", "ip", 16.92, null, 17.71, 1400],
      ["HR", "hr", 2.5012, null, 2.8877, 1000],
    ];
    for (const [title, metric, model, league, prev, n] of expected) {
      const text = cardFor(title).textContent ?? "";
      expect(text, title).toContain(`${fmtCount(n)}명`);
      if (league === null) {
        expect(text, `${title}: 출전량·HR은 리그 평균 막대가 없다`).not.toContain("리그 평균");
        expectInOrder(text, ["이 모델", fmtMae(metric, model), "직전 시즌", fmtMae(metric, prev)]);
      } else {
        expectInOrder(text, [
          "이 모델",
          fmtMae(metric, model),
          "리그 평균",
          fmtMae(metric, league),
          "직전 시즌",
          fmtMae(metric, prev),
        ]);
      }
    }
  });

  it("가중 모델 값(.0671·.902)은 카드에 쓰지 않는다 — 확장이 있으면 확장이 최종 모델", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();
    expect(cardFor("OPS").textContent).not.toContain(".0671");
    expect(cardFor("ERA").textContent).not.toContain(".902");
  });

  it("투수에 확장 모델 행이 없으면 가중 모델이 최종 모델이다", async () => {
    const report = handMadeReport();
    report.rows = report.rows.filter((r) => !(r.role === "pit" && r.model === "extended"));
    serve(report);
    renderPage();
    await loaded();
    expectInOrder(cardFor("ERA").textContent ?? "", ["이 모델", ".902", "리그 평균", ".956"]);
    expectInOrder(cardFor("OPS").textContent ?? "", ["이 모델", ".0655"]);
  });

  it("요약 문장: 여섯 지표 모두 이겼다는 말과, 진 그룹(1~2시즌 HR, 3~5시즌 IP)을 숫자와 함께", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();
    expect(screen.getAllByText(/여섯 지표 모두/).length).toBeGreaterThan(0);
    expectSentenceWith(["1~2시즌", "HR", fmtMae("hr", 1.8123), fmtMae("hr", 1.7004)]);
    expectSentenceWith(["3~5시즌", "IP", fmtMae("ip", 19.87), fmtMae("ip", 19.12)]);
    // 이긴 그룹 값은 진 그룹으로 적지 않는다 (1~2시즌 PA 45.1 대 47.0)
    expect(innermostWith(["1~2시즌", fmtMae("pa", 45.12), fmtMae("pa", 47.03)])).toHaveLength(0);
    expect(screen.getAllByText(/낮을수록 좋/).length).toBeGreaterThan(0);
  });

  it("전체 그룹에서 진 지표가 있으면 '여섯 지표 모두'라고 하지 않고 그 지표를 적는다", async () => {
    const report = handMadeReport();
    report.rows = report.rows.map((r) =>
      r.target_season === 2026 && r.role === "pit" && r.model === "extended" && r.metric === "era" && r.group === "all"
        ? { ...r, mae: 0.9701 }
        : r,
    );
    serve(report);
    renderPage();
    await loaded();
    expect(screen.queryAllByText(/여섯 지표 모두/)).toHaveLength(0);
    // 카드(이 모델·리그 평균 막대)가 아니라 요약 문장에서 찾는다
    expectSentenceWith(["ERA", "리그 평균", fmtMae("era", 0.9701), fmtMae("era", 0.9562)], ["이 모델"]);
  });

  it("다시 돌린 이유 상자: 다시 돌림 횟수, 사유, 첫 결과 숫자 (D31)", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();
    expect(screen.getAllByText(/다시 돌림 1번/).length).toBeGreaterThan(0);
    const box = innermostWith(["두 번 돌린 이유", REASON])[0];
    expect(box, "이유 상자").toBeDefined();
    const text = box!.textContent ?? "";
    expect(text).toContain("첫 결과");
    // 첫 실행에서 달랐던 지표의 숫자
    for (const [metric, v] of [
      ["ops", 0.0701],
      ["k_bb_pct", 3.911],
      ["era", 0.9123],
      ["hr", 2.4449],
    ] as const) {
      expect(text, metric).toContain(fmtMae(metric, v));
    }
    expect(box!.querySelector("table")).toBeNull();
  });

  it("다시 돌린 적이 없으면 이유 상자가 없다", async () => {
    const report = handMadeReport();
    report.final_test = { ...report.final_test!, reruns: [], first_run: null };
    serve(report);
    renderPage();
    await loaded();
    expect(screen.queryAllByText(/두 번 돌린 이유/)).toHaveLength(0);
    expect(screen.queryAllByText(REASON)).toHaveLength(0);
    expectInOrder(cardFor("OPS").textContent ?? "", ["이 모델", ".0655"]);
  });
});

describe("검증 화면 — ± 범위 표 (D29)", () => {
  it("지표 × 경력 그룹(1~2시즌·3~5시즌), 값은 fmtBand, 대상 수", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();
    const table = tableWithHeader(/1~2시즌/);
    const headers = within(table).getAllByRole("columnheader").map((h) => h.textContent ?? "");
    expect(headers.some((h) => h.includes("3~5시즌"))).toBe(true);
    const cases: Array<[string, BandRow["metric"], number, number, number, number]> = [
      ["OPS", "ops", 0.111, 0.099, 250, 700],
      ["ERA", "era", 1.52, 1.41, 260, 610],
      ["K−BB%", "k_bb_pct", 6.1, 5.7, 260, 610],
    ];
    for (const [label, metric, short, long, nShort, nLong] of cases) {
      const found = rowsWith(table, [label, fmtBand(metric, short), fmtBand(metric, long)]);
      expect(found, label).toHaveLength(1);
      expectInOrder(found[0]!.textContent ?? "", [
        fmtBand(metric, short),
        fmtBand(metric, long),
        `${fmtCount(nShort)}명`,
        `${fmtCount(nLong)}명`,
      ]);
    }
  });
});

describe("검증 화면 — 모델별·지표별 표", () => {
  it("선택 상자 3개: 목표 시즌(행에서, 최신 먼저)·역할·그룹(한국어 이름)", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();
    expect(await optionsOf("목표 시즌")).toEqual(["2026", "2025", "2024"]);
    expect(await optionsOf("역할")).toEqual(["타자", "투수"]);
    const groups = await optionsOf("그룹");
    for (const g of ["전체", "1~2시즌", "3~5시즌", "직전 시즌 출전", "직전 시즌 미출전"]) {
      expect(groups, g).toContain(g);
    }
  });

  it("기본은 최신 목표 시즌·타자·전체: 모델·지표·대상 수·평균 오차", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();
    const table = tableWithHeader(/평균 오차/);
    const headers = within(table).getAllByRole("columnheader").map((h) => (h.textContent ?? "").trim());
    for (const h of ["모델", "지표", "대상 수", "평균 오차"]) expect(headers, h).toContain(h);
    expect(rowsWith(table, ["확장", "OPS", "300", ".0655"])).toHaveLength(1);
    expect(rowsWith(table, ["가중", "OPS", ".0671"])).toHaveLength(1);
    expect(rowsWith(table, ["리그 평균", "OPS", ".0744"])).toHaveLength(1);
    expect(rowsWith(table, ["직전 시즌", "OPS", ".0912"])).toHaveLength(1);
    expect(rowsWith(table, ["확장", "PA", "1,000", "70.0"])).toHaveLength(1);
    expect(rowsWith(table, ["확장", "HR", "2.50"])).toHaveLength(1);
    // 투수 지표·다른 시즌·다른 그룹은 없다
    expect(rowsWith(table, ["ERA"])).toHaveLength(0);
    expect(rowsWith(table, [".0688"])).toHaveLength(0);
    expect(rowsWith(table, [".0702"])).toHaveLength(0);
  });

  it("역할을 투수로 바꾸면 투수 지표", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();
    await choose("역할", "투수");
    const table = tableWithHeader(/평균 오차/);
    expect(rowsWith(table, ["확장", "ERA", "280", ".881"])).toHaveLength(1);
    expect(rowsWith(table, ["확장", "K−BB%", "3.71"])).toHaveLength(1);
    expect(rowsWith(table, ["확장", "IP", "1,400", "16.9"])).toHaveLength(1);
    expect(rowsWith(table, ["OPS"])).toHaveLength(0);
  });

  it("목표 시즌을 2025로 바꾸면 2025 값", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();
    await choose("목표 시즌", "2025");
    const table = tableWithHeader(/평균 오차/);
    expect(rowsWith(table, ["확장", "OPS", "310", ".0688"])).toHaveLength(1);
    expect(rowsWith(table, [".0655"])).toHaveLength(0);
  });

  it("그룹을 1~2시즌으로 바꾸면 그 그룹 값", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();
    await choose("그룹", "1~2시즌");
    const table = tableWithHeader(/평균 오차/);
    expect(rowsWith(table, ["확장", "HR", "400", "1.81"])).toHaveLength(1);
    expect(rowsWith(table, ["확장", "OPS", "90", ".0702"])).toHaveLength(1);
    expect(rowsWith(table, [".0655"])).toHaveLength(0);
  });

  it("평가 집단 설명(200타석·50이닝, 출전량은 전원)과 방법 설명 링크", async () => {
    serve(handMadeReport());
    renderPage();
    await loaded();
    expectSentenceWith(["200타석", "50이닝"]);
    expect(screen.getAllByText(/예측 대상 전원/).length).toBeGreaterThan(0);
    // next/link는 테스트에서 trailingSlash 설정을 모르므로 끝 / 는 있어도 없어도 된다
    expect(screen.getByRole("link", { name: /방법 설명/ }).getAttribute("href")).toMatch(/^\/about\/?$/);
  });
});

describe("검증 화면 — 2026 최종 검증 전 (final_test null, 합성 샘플)", () => {
  it("2026 시험을 아직 돌리지 않았다고 적고, 2025 표·± 범위 표를 보여 준다", async () => {
    serve(synthetic);
    renderPage();
    await loaded();
    expectSentenceWith(["2026", "아직"]);
    expect(screen.queryAllByText("이 모델")).toHaveLength(0);
    expect(screen.queryAllByText(/두 번 돌린 이유/)).toHaveLength(0);
    expect(await optionsOf("목표 시즌")).toEqual(["2025"]);
    const table = tableWithHeader(/평균 오차/);
    expect(rowsWith(table, ["가중", "OPS", "412", fmtMae("ops", 0.066)])).toHaveLength(1);
    expect(rowsWith(table, ["가중", "PA", "1,520", fmtMae("pa", 131)])).toHaveLength(1);
    const bands = tableWithHeader(/3~5시즌/);
    expect(rowsWith(bands, ["OPS", fmtBand("ops", 0.097)])).toHaveLength(1);
  });
});
