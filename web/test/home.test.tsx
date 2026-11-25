// 첫 화면 히어로·고지·공통 머리글/바닥글 (DESIGN SA-1, SPEC D16·D19·D27, N10) — 레인 A Red 테스트
// 예측표 동작은 projections.test.tsx가 본다. 데이터는 가짜 fetch로 준다(네트워크 없음).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import HomePage from "@/app/page";
import { Providers } from "@/app/providers";
import { DATA_BASE } from "@/src/data/load";
import type { BatTable, Meta, PitTable } from "@/src/data/types";
import { fmtCount } from "@/src/format";
import { PHOTO_CREDIT } from "@/src/photo";
import { DISCLAIMER_SHORT, SOURCES_LINE } from "@/src/site";

const ROOT = join(import.meta.dirname, "..", "..");
const sample = <T,>(name: string): T =>
  JSON.parse(readFileSync(join(ROOT, "tests/fixtures/synthetic/web", name), "utf-8")) as T;
const sampleBat = sample<BatTable>("batters.json");
const samplePit = sample<PitTable>("pitchers.json");

// 입력 시즌을 실데이터와 다르게 둬서 화면이 meta에서 읽는지 본다
const META: Meta = {
  ...sampleBat.meta,
  input_seasons: [2021, 2022, 2023, 2024, 2025],
  disclaimer:
    "재미로 보는 예측입니다. 최근 5시즌 기록으로 만든 통계 모델의 점 예측이며, 실제 성적을 보장하지 않습니다.",
};

/** 합성 샘플 행을 늘려 대상 수가 천 단위가 되게 한다 (칩 숫자는 행 수에서 센다) */
function manyBatters(count: number): BatTable {
  const base = sampleBat.rows[0]!;
  return {
    ...sampleBat,
    meta: META,
    rows: Array.from({ length: count }, (_, i) => ({
      ...base,
      player: { ...base.player, player_id: 10_000 + i, name: `Batter ${i + 1}` },
      projection: { ...base.projection, ops: 1 - i * 0.0004 },
    })),
  };
}

function manyPitchers(count: number): PitTable {
  const base = samplePit.rows[0]!;
  return {
    ...samplePit,
    meta: META,
    rows: Array.from({ length: count }, (_, i) => ({
      ...base,
      player: { ...base.player, player_id: 20_000 + i, name: `Pitcher ${i + 1}` },
      projection: { ...base.projection, era: 2 + i * 0.002 },
      prev: { season: 2026, ip: 50, era: 3.5, k_bb_pct: 15 },
    })),
  };
}

const BATTERS = manyBatters(1144);
const PITCHERS = manyPitchers(1490);
const BAT_URL = `${DATA_BASE}/batters.json`;
const PIT_URL = `${DATA_BASE}/pitchers.json`;

function stubData(files: Record<string, unknown>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
    const raw = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const body = files[raw.replace(/^https?:\/\/[^/]+/, "")];
    return body === undefined
      ? new Response("not found", { status: 404 })
      : new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  // 데스크톱 너비 — min-width 질의만 맞는 것으로
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: /min-width/.test(query) && !/max-width/.test(query),
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  }));
  stubData({ [BAT_URL]: BATTERS, [PIT_URL]: PITCHERS });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  // 예측표는 보기 상태를 주소 해시에 적는다 — 다음 테스트로 새지 않게 주소를 되돌린다
  window.history.replaceState(null, "", "/");
});

const norm = (s: string | null | undefined): string => (s ?? "").replace(/\s+/g, " ").trim();

/** 글자가 여러 요소로 나뉘어 있어도(<b>1,144</b>명) 그 글을 담은 가장 안쪽 요소를 찾는 매처 */
function textMatching(re: RegExp) {
  return (_content: string, el: Element | null): boolean => {
    if (el === null || !re.test(norm(el.textContent))) return false;
    return Array.from(el.children).every((c) => !re.test(norm(c.textContent)));
  };
}

function renderHome() {
  return render(
    <Providers>
      <HomePage />
    </Providers>,
  );
}

const findTable = () => screen.findByRole("table", {}, { timeout: 5000 });

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

describe("첫 화면 히어로 (SA-1, DS5)", () => {
  it("배경 사진(/images/hero.jpg, 장식), 머리말, 큰 제목, 한 줄 소개", async () => {
    renderHome();
    await findTable();
    // 사진은 <img alt="">(장식)이든 배경 스타일이든 된다
    const imgs = Array.from(document.querySelectorAll("img")).filter((img) =>
      decodeURIComponent(`${img.getAttribute("src") ?? ""} ${img.getAttribute("srcset") ?? ""}`).includes(
        "/images/hero.jpg",
      ),
    );
    const styled = Array.from(document.querySelectorAll("[style]")).filter((el) =>
      (el.getAttribute("style") ?? "").includes("/images/hero.jpg"),
    );
    expect(imgs.length + styled.length, "히어로 사진").toBeGreaterThan(0);
    for (const img of imgs) expect(img).toHaveAttribute("alt", "");

    expect(screen.getAllByText(textMatching(/PROJECTIONS/)).length).toBeGreaterThan(0);
    expect(screen.getAllByText(textMatching(/JUST FOR FUN/i)).length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("2027 MLB 선수 성적 예측");
    expect(screen.getAllByText(textMatching(/최근 5시즌 기록으로 계산한/)).length).toBeGreaterThan(0);
  }, 20000);

  it("대상 수 칩은 불러온 표의 행 수 — 타자 1,144명, 입력 2021~2025; 투수는 투수 표를 받은 뒤 1,490명", async () => {
    renderHome();
    await findTable();
    expect(
      screen.getAllByText(textMatching(new RegExp(`타자\\s*${fmtCount(BATTERS.rows.length)}\\s*명`))).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText(textMatching(/입력\s*2021\s*~\s*2025/)).length).toBeGreaterThan(0);

    // 투수 표는 처음 투수 탭을 누를 때 받는다 (projections.test.tsx) — 받은 뒤 투수 칩이 실제 수를 보인다
    fireEvent.click(roleTab("투수"));
    await waitFor(
      () =>
        expect(
          screen.getAllByText(textMatching(new RegExp(`투수\\s*${fmtCount(PITCHERS.rows.length)}\\s*명`))).length,
        ).toBeGreaterThan(0),
      { timeout: 5000 },
    );
  }, 20000);

  it("단추: '예측표 보기'는 표로 가는 앵커, '얼마나 맞았나'는 /validation/", async () => {
    renderHome();
    const table = await findTable();
    const toTable = screen.getByRole("link", { name: /예측표 보기/ });
    const href = toTable.getAttribute("href") ?? "";
    expect(href).toMatch(/^#.+/);
    const target = document.getElementById(decodeURIComponent(href.slice(1)));
    expect(target, `앵커 대상 ${href}`).not.toBeNull();
    expect(target!.contains(table)).toBe(true);

    // next/link는 테스트에서 끝의 /를 뗀다 — 빌드(trailingSlash)는 /validation/
    const toValidation = screen.getByRole("link", { name: /얼마나 맞았나/ });
    expect(toValidation.getAttribute("href")).toMatch(/^\/validation\/?$/);
  }, 20000);

  it("표 위 고지 한 줄 (D16) — 바닥글과 별도로, 역할 탭보다 앞", async () => {
    renderHome();
    const table = await findTable();
    const footer = document.querySelector("footer");
    expect(footer).not.toBeNull();
    const lines = screen.getAllByText(textMatching(/보장하지 않/)).filter((el) => !footer!.contains(el));
    expect(lines.length, "본문 고지").toBeGreaterThan(0);
    const tab = roleTab("타자");
    const precedes = (a: Element, b: Element) => (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0;
    expect(lines.some((line) => precedes(line, tab) && precedes(line, table))).toBe(true);
  }, 20000);
});

describe("공통 머리글·바닥글 (F16, D16·D27)", () => {
  it("예측 메뉴가 현재 화면이고, 바닥글에 고지·출처·사진 저작권", async () => {
    renderHome();
    await findTable();
    const nav = screen.getByRole("navigation", { name: "주요 메뉴" });
    expect(within(nav).getByRole("link", { current: "page" })).toHaveTextContent("예측");
    const footer = document.querySelector("footer")!;
    expect(footer).toHaveTextContent(DISCLAIMER_SHORT);
    expect(footer).toHaveTextContent(SOURCES_LINE);
    expect(footer).toHaveTextContent(PHOTO_CREDIT);
  }, 20000);

  it("제목 단계가 건너뛰지 않는다 — h1 하나로 시작 (N10)", async () => {
    renderHome();
    await findTable();
    const levels = screen
      .getAllByRole("heading")
      .map((h) => Number(h.getAttribute("aria-level") ?? h.tagName.replace(/^H/, "")));
    expect(levels[0]).toBe(1);
    expect(levels.filter((l) => l === 1)).toHaveLength(1);
    levels.slice(1).forEach((level, i) => {
      expect(level, `제목 단계 ${levels.join(",")}`).toBeLessThanOrEqual(levels[i]! + 1);
    });
  }, 20000);
});
