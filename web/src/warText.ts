// WAR 화면 공통 문구 — 합의 시안(DESIGN §4-1 SA-6~9)과 SPEC D33·D40·D41·D42·E28·E29 그대로 (메인 세션만 고친다)
// 화면마다 문구를 따로 쓰지 않고 여기서 가져다 쓴다. 제작기 영상도 같은 설명을 쓴다(D40).

export const WAR_LABEL = "WAR(MLB Stats API 기준)";

/** 순위표 아래 한 줄 (SA-6) */
export const WAR_ABOUT =
  "WAR(MLB Stats API 기준) — 대체 선수보다 몇 승을 더 만들었나. fWAR·bWAR과 값이 다를 수 있습니다. 투타 겸업은 타자 WAR + 투수 WAR.";

/** `*` 표시의 뜻 (D30) */
export const WAR_CONSERVATIVE = "예측 WAR과 예측 출전은 부상·결장 위험을 포함한 보수적 예상입니다.";

/** D42 — war.json은 2026 출전 선수만 싣는다 */
export const WAR_PREV_ONLY = "WAR 순위는 2026 출전 선수만 싣습니다.";
/** 2026에 뛰지 않은 선수의 상세 (DESIGN §4-1 표시 규칙) */
export const WAR_NOT_IN_FILE = "WAR 예측은 2026 출전 선수만 보여 줍니다.";

/** D41 */
export const FULL_TIME_NAME = "풀타임 환산 WAR";
export const FULL_TIME_SHORT = "풀타임 환산";
export const FULL_TIME_ADJUSTED = "출전량을 맞춘 조정 WAR";
/** 기준 출전량 — "타자 600타석 · 선발 180이닝 · 불펜 65이닝" (숫자는 war.json full_time에서) */
export function fullTimeUnits(full: { pa: number; sp_ip: number; rp_ip: number }): string {
  return `타자 ${full.pa}타석 · 선발 ${full.sp_ip}이닝 · 불펜 ${full.rp_ip}이닝`;
}
/** 기준 출전량 문구 — 끝에 " 기준" */
export function fullTimeBasis(full: { pa: number; sp_ip: number; rp_ip: number }): string {
  return `${fullTimeUnits(full)} 기준`;
}

/** 「WAR은 어떻게 계산되나」 첫 문단 (SA-8) */
export const WAR_WHAT =
  "WAR(대체 선수 대비 승리 기여)은 「흔히 구할 수 있는 대체 선수보다 몇 승을 더 만들었나」입니다. 이 사이트는 MLB Stats API의 WAR을 씁니다 — FanGraphs fWAR과 같은 틀이지만 값이 조금 다를 수 있습니다.";

/** 타자 WAR 구성 항목 (E29 — 점 단위, 원본 그대로). note는 일반 설명 */
export const BAT_RUNS_PARTS = [
  { key: "runs_batting", name: "타격", note: "출루·장타로 만든 점수" },
  { key: "runs_baserunning", name: "주루", note: "도루·진루" },
  { key: "runs_fielding", name: "수비", note: "수비 범위·송구 등" },
  { key: "runs_positional", name: "포지션 보정", note: "어려운 자리는 더하고, 지명타자는 크게 뺀다" },
  { key: "runs_league", name: "리그 보정", note: "리그 간 차이" },
  { key: "runs_replacement", name: "대체 수준", note: "대체 선수 대비 기본 점수" },
] as const;

export const BAT_WAR_SUMMARY =
  "타격·주루만이 아니다. 유격수·포수처럼 어려운 자리는 더하고, 지명타자는 크게 뺀다.";

/** 점 → 승 (E29): 선수별 비율이 아니라 그 시즌 리그 비율로만 적는다 */
export function runsPerWinNote(rpw: { season: number; value: number }): string {
  return `${rpw.season}년은 약 ${rpw.value.toFixed(1)}점이 1승`;
}
export const RUNS_NOT_EXACT = "구성 항목은 원본 값 그대로 — 합이 WAR과 정확히 비례하지 않는 선수도 있습니다.";

/** 투수 WAR 정의 (D40 · E28) */
export const PIT_WAR_TITLE = "투수 WAR — FIP 기반";
export const PIT_WAR_FIP_COL = "FIP 기반 (예측에 씀)";
export const PIT_WAR_RA9_COL = "실점 기반 (참고)";
export const PIT_WAR_NOTE =
  "FIP 기반은 삼진·볼넷·홈런으로, 실점 기반은 실제 내준 점수로 계산합니다. 실점은 수비·운에 많이 흔들려서, 다음 해 예측에는 FIP 기반이 더 안정적이었습니다.";
/** 투수 WAR 정의별 작년 → 올해 상관 (2024 · 2025, 주전급 150이닝 이상 — SPEC E28 실측) */
export const PIT_WAR_STABILITY = {
  caption: "작년 → 올해 상관 (2024 · 2025)",
  rows: [
    { name: "FIP 기반 (이 사이트)", corr: [0.5, 0.62], isUsed: true },
    { name: "FIP · 실점 절반씩", corr: [0.4, 0.63], isUsed: false },
    { name: "실점 기반", corr: [0.22, 0.55], isUsed: false },
  ],
  note: "실점은 수비·운에 많이 흔들린다. 다음 해를 예측하는 사이트라 가장 안정적인 FIP 기반을 쓴다(주전급 150이닝 이상).",
} as const;

/** 검증 화면 카드 셋 (SA-8) */
export const WAR_FACTS = [
  {
    title: "1,000승",
    body: "한 시즌 리그 전체 WAR — 타자 570 + 투수 430. 2019~2026 여덟 시즌 모두 정확히 맞아, 데이터를 다 받았는지 검사하는 데 쓴다.",
  },
  {
    title: "예측 = 비율 × 출전량",
    body: "최근 5시즌 타석당·아웃당 WAR을 최근일수록 크게 더하고 리그 평균 쪽으로 당긴 뒤, 예측 출전량을 곱한다. 투타 겸업은 두 역할을 더한다.",
  },
  {
    title: "「조정 WAR」은?",
    body: "표준 지표는 없다. OPS+·ERA+ 같은 조정 지표의 구장·리그 보정은 WAR 안에 이미 들어 있다. 이 사이트의 풀타임 환산 WAR은 출전량만 맞춘 값이다.",
  },
] as const;

/** 2026 시험 (SA-8) */
export const WAR_TEST_INTRO =
  "WAR 설정은 2024·2025 시즌으로만 정하고 커밋한 뒤, 2026 시즌을 한 번만 시험했습니다. 숫자는 평균 오차(승)이고 낮을수록 좋습니다.";

/** 실제 WAR 2 이상 칸에서 졌을 때의 해석 (SA-8 · SPEC E27 · E28 — 2024·2025 실측) */
export const WAR_STAR_LOSS_REASON =
  "실제 WAR 2 이상인 선수에서 지는 것은 출전량을 보수적으로 잡는 약점이 그대로 나타난 것입니다(2024·2025 과거 검증에서는 타자·투수 모두 작년 그대로에 졌습니다).";
