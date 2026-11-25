// 소개 화면 (SA-5) — 레인 C Red 테스트. 정적 글이라 데이터를 받지 않는다.
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as aboutModule from "@/app/about/page";
import { Providers } from "@/app/providers";
import { DISCLAIMER_SHORT, REPO_URL } from "@/src/site";

const AboutPage = aboutModule.default;

function renderPage() {
  return render(
    <Providers>
      <AboutPage />
    </Providers>,
  );
}

const bodyText = () => document.body.textContent ?? "";

/** 글자들이 문서에 이 순서로 나오는지 */
function expectInOrder(parts: string[]) {
  const text = bodyText();
  let at = -1;
  for (const p of parts) {
    const next = text.indexOf(p, at + 1);
    expect(next, `"${p}"가 앞 조각 뒤에 있어야 한다`).toBeGreaterThan(at);
    at = next;
  }
}

/** 조각을 모두 담은 가장 안쪽 요소들 (한 문장·한 단락) */
function innermostWith(parts: string[]): HTMLElement[] {
  const ok = (el: Element) => parts.every((p) => (el.textContent ?? "").includes(p));
  return Array.from(document.body.querySelectorAll<HTMLElement>("*")).filter(
    (el) => ok(el) && !Array.from(el.children).some(ok),
  );
}

function expectSentenceWith(parts: string[]) {
  const found = innermostWith(parts).filter((el) => (el.textContent ?? "").length < 400);
  expect(found.length, `한 문장에 ${parts.join(" · ")}`).toBeGreaterThan(0);
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  // 소개는 정적 글 — 데이터를 받으면 안 된다
  fetchMock = vi.fn(() => Promise.reject(new Error("소개 화면은 네트워크를 쓰지 않는다")));
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("소개 화면 — 공통·머리", () => {
  it("헤더(소개 선택)·바닥글, 제목, 리드 문구, 데이터를 받지 않는다", () => {
    renderPage();
    expect(screen.getByRole("link", { current: "page" })).toHaveTextContent("소개");
    expect(screen.getByText(DISCLAIMER_SHORT)).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("이 예측은 어떻게 만들었나");
    expectSentenceWith(["최근 5시즌(2022~2026)", "과거 기록만으로"]);
    expectSentenceWith(["팬이 재미로 만든 예측", "구단·리그와 관계가 없습니다"]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("제목 단계가 차례대로고, 섹션은 방법 → 어떻게 검증했나 → 데이터 → 알아 두실 점 → 고지 순서", () => {
    renderPage();
    const headings = screen.getAllByRole("heading");
    const levels = headings.map((h) => Number(h.getAttribute("aria-level") ?? h.tagName.slice(1)));
    expect(levels[0]).toBe(1);
    expect(levels.filter((l) => l === 1)).toHaveLength(1);
    levels.forEach((l, i) => {
      if (i > 0) expect(l, `제목 단계가 건너뛴다: ${levels.join(",")}`).toBeLessThanOrEqual(levels[i - 1]! + 1);
    });
    const h2 = screen.getAllByRole("heading", { level: 2 }).map((h) => (h.textContent ?? "").trim());
    const order = ["방법", "어떻게 검증했나", "데이터", "알아 두실 점"];
    const idx = order.map((name) => h2.indexOf(name));
    expect(idx, `h2: ${h2.join(" / ")}`).not.toContain(-1);
    expect([...idx].sort((a, b) => a - b)).toEqual(idx);
    // 고지는 마지막 섹션 (제목이든 굵은 글이든 "고지"가 알아 두실 점 뒤에)
    expectInOrder(["알아 두실 점", "고지", "MLB Advanced Media"]);
  });

  it("페이지 제목(metadata)에 '소개'가 들어간다 — 레이아웃 제목 틀(%s)과 함께 쓴다", () => {
    const meta = (aboutModule as unknown as { metadata?: { title?: unknown } }).metadata;
    expect(meta, "app/about/page.tsx가 metadata를 내보낸다").toBeDefined();
    expect(JSON.stringify(meta!.title)).toContain("소개");
  });
});

describe("소개 화면 — 방법 (SA-5)", () => {
  it("4단계가 순서대로, SA-5 문구 그대로", () => {
    renderPage();
    expectInOrder([
      "최근 시즌일수록 크게",
      "리그 평균 쪽으로 당기기",
      "출전량은 따로",
      "나이·Statcast로 다듬기",
    ]);
    expectSentenceWith(["직전 시즌에 가장 큰 무게", "8·4·2·1·0.5"]);
    expectSentenceWith(["기록이 적을수록 리그 평균에 가깝게"]);
    expectSentenceWith(["직전 출전량, 공백, 선발 비중"]);
    expectSentenceWith(["헛스윙률", "HR·OPS·ERA·K−BB%"]);
  });

  it("섹션 이미지 /images/section-method.jpg는 장식이다 (alt=\"\")", () => {
    const { container } = renderPage();
    const img = container.querySelector<HTMLImageElement>('img[src*="/images/section-method.jpg"]');
    expect(img, "섹션 이미지").not.toBeNull();
    expect(img!.getAttribute("alt")).toBe("");
    // 방법 제목 뒤, 첫 단계 앞에 있다
    const methodHeading = screen.getByRole("heading", { level: 2, name: "방법" });
    const firstStep = screen.getAllByText(/최근 시즌일수록 크게/)[0]!;
    expect(methodHeading.compareDocumentPosition(img!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(img!.compareDocumentPosition(firstStep) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // 모든 이미지에 alt 속성이 있다 (N10)
    for (const each of Array.from(container.querySelectorAll("img"))) {
      expect(each.hasAttribute("alt"), each.outerHTML).toBe(true);
    }
  });
});

describe("소개 화면 — 검증·데이터·알아 두실 점·고지", () => {
  it("검증 절차 3단계(다시 돌림 포함)와 검증 결과 링크", () => {
    renderPage();
    expectInOrder([
      "2024·2025 시즌을 맞혀 보며 정했습니다",
      "2026 시즌을 한 번만 시험했습니다",
      "2026 시험을 한 번 더 돌렸고, 두 결과를 모두 공개합니다",
    ]);
    const link = screen.getByRole("link", { name: /검증 결과 보기/ });
    // next/link는 테스트에서 trailingSlash 설정을 모르므로 끝 / 는 있어도 없어도 된다
    expect(link.getAttribute("href")).toMatch(/^\/validation\/?$/);
  });

  it("데이터: 출처 두 곳과 내용, 기간(2019~2026, 2020은 60경기), 만든 시점", () => {
    renderPage();
    expectSentenceWith(["MLB Stats API", "시즌 기록, 선수 정보"]);
    expectSentenceWith(["Baseball Savant", "Statcast 지표, OAA, Fielding Run Value"]);
    expectSentenceWith(["2019~2026 정규 시즌", "2020년은 60경기"]);
    expectSentenceWith(["2026 시즌 종료 기준"]);
  });

  it("알아 두실 점 4가지: 출전량 보수적(D30) · 범위가 넓음(D29) · 신인 없음 · 팀 이동 미반영", () => {
    renderPage();
    expectSentenceWith(["출전량(PA·IP)과 HR", "보수적 예상"]);
    expectSentenceWith(["± 범위는 넓습니다"]);
    expectSentenceWith(["MLB 기록이 없는 선수", "신인", "예측하지 않습니다"]);
    expectSentenceWith(["팀 이동", "반영하지 않습니다"]);
    expectInOrder(["보수적 예상", "± 범위는 넓습니다", "MLB 기록이 없는 선수", "팀 이동"]);
  });

  it("목록 문장은 잘리지 않는다 — ListItem에 문자열이 아니라 Text로 넘긴다", () => {
    // Astryx ListItem은 문자열 label(조각으로 감싼 문자열 포함)을 한 줄로 자르고 말줄임표를 붙인다.
    // 휴대폰 폭에서 데이터·알아 두실 점 문장이 "…"로 끊겼다 — 문장 전체가 Text(.astryx-text) 하나에 들어 있어야 줄바꿈된다.
    renderPage();
    const items = within(screen.getByRole("main")).getAllByRole("listitem");
    expect(items.length, "검증 3 + 데이터 4 + 알아 두실 점 4").toBe(11);
    for (const li of items) {
      const full = (li.textContent ?? "").trim();
      const wrapped = Array.from(li.querySelectorAll(".astryx-text")).some(
        (el) => (el.textContent ?? "").trim() === full,
      );
      expect(wrapped, `문장 전체가 Text 안에: ${full}`).toBe(true);
    }
  });

  it("고지 상자: 재미로 보는 예측·베팅 조언 아님, MLBAM 권리(자료·선수 사진), 깃허브 링크", () => {
    renderPage();
    expectSentenceWith(["베팅·판타지 조언이 아닙니다"]);
    expectSentenceWith(["선수 사진", "MLB Advanced Media, L.P."]);
    // 바닥글의 "소스 (GitHub)"와 고지 안의 링크, 둘 다 저장소로
    const repoLinks = screen.getAllByRole("link").filter((a) => a.getAttribute("href") === REPO_URL);
    expect(repoLinks.length).toBeGreaterThanOrEqual(2);
    const footer = screen.getByRole("contentinfo");
    expect(repoLinks.filter((a) => !footer.contains(a)).length).toBeGreaterThanOrEqual(1);
    expect(within(footer).getByText(DISCLAIMER_SHORT)).toBeInTheDocument();
  });
});
