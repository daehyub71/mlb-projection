// 사이트 공통 상수 (DESIGN v1.0 §6) — 메인 세션만 고친다

export type NavKey = "projections" | "validation" | "about";

export const NAV: ReadonlyArray<{ key: NavKey; href: string; label: string }> = [
  { key: "projections", href: "/", label: "예측" },
  { key: "validation", href: "/validation/", label: "검증" },
  { key: "about", href: "/about/", label: "소개" },
];

export const SITE_NAME = "MLB 2027 예측";
export const SITE_TITLE = "재미로 보는 2027 MLB 선수 성적 예측";
export const SITE_DESCRIPTION = "최근 5시즌 기록으로 만든 2027 MLB 타자·투수 성적 예측 (재미로 보는 예측)";
export const DISCLAIMER_SHORT = "재미로 보는 예측 — 실제 성적을 보장하지 않습니다";
export const SOURCES_LINE = "출처: MLB Stats API, Baseball Savant";
export const REPO_URL = "https://github.com/daehyub71/mlb-projection";

/** 예측표 한 쪽의 행 수 (DS4) */
export const PAGE_SIZE = 50;

/** 선수 상세 주소 (DS1) */
export function playerHref(playerId: number): string {
  return `/player/?id=${playerId}`;
}

/** 표 아래 설명 (D29·D30) */
export const BAND_NOTE = "± 과거 오차 80% 범위 — 비슷한 경력 선수 10명 중 8명은 실제 성적이 이 안에 들어왔습니다.";
export const PLAYING_TIME_NOTE = "* 출전량(PA·IP)과 HR은 부상·결장 위험을 포함한 보수적 예상입니다. 주전 선수는 평소 출전량보다 낮게 나오는 편입니다.";

/** 경력 그룹 이름 (D13) */
export const CAREER_LABEL = { seasons_1_2: "1~2시즌", seasons_3_5: "3~5시즌" } as const;
export function careerGroup(nSeasons: number): "seasons_1_2" | "seasons_3_5" {
  return nSeasons <= 2 ? "seasons_1_2" : "seasons_3_5";
}
