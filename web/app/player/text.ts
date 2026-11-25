// 선수 상세(SA-2) 화면 글자 만들기 — 숫자는 모두 src/format.ts를 거친다
import type {
  BandMetric,
  BatDetail,
  GroupName,
  InputWindow,
  Metric,
  PitDetail,
  Role,
} from "@/src/data/types";
import { DASH, fmtBand, fmtCount, fmtDiff, fmtIp, fmtMae, fmtMetric, fmtPct } from "@/src/format";
import { CAREER_LABEL, careerGroup } from "@/src/site";

export type RoleDetail = { role: "bat"; detail: BatDetail } | { role: "pit"; detail: PitDetail };

export const ROLE_NOUN: Record<Role, string> = { bat: "타자", pit: "투수" };

// 포지션 이름 — 계약에 표가 없어 화면에서 둔다 (모르는 코드는 그대로)
const POSITION_LABEL: Record<string, string> = {
  C: "포수",
  "1B": "1루수",
  "2B": "2루수",
  "3B": "3루수",
  SS: "유격수",
  LF: "좌익수",
  CF: "중견수",
  RF: "우익수",
  OF: "외야수",
  DH: "지명타자",
  P: "투수",
  TWP: "투타 겸업",
};

const BATS_LABEL: Record<string, string> = { R: "우타", L: "좌타", S: "양타" };
const THROWS_LABEL: Record<string, string> = { R: "우투", L: "좌투", S: "양투" };

/** "SS" → "유격수(SS)", "TWP" → "투타 겸업(TWP)". 없으면 null */
export function positionText(code: string | null): string | null {
  if (code == null || code === "") return null;
  const label = POSITION_LABEL[code];
  return label === undefined ? code : `${label}(${code})`;
}

/** 투타: "좌타/우투". 한쪽만 알면 그쪽만, 둘 다 모르면 null */
export function handednessText(bats: string | null, throws: string | null): string | null {
  const parts = [
    bats == null ? null : (BATS_LABEL[bats] ?? bats),
    throws == null ? null : (THROWS_LABEL[throws] ?? throws),
  ].filter((p): p is string => p !== null);
  return parts.length === 0 ? null : parts.join("/");
}

/** "2022~2026 · 5시즌" (한 시즌이면 "2025 · 1시즌") */
export function windowText(w: InputWindow): string {
  const span = w.input_first === w.input_last ? String(w.input_first) : `${w.input_first}~${w.input_last}`;
  return `${span} · ${fmtCount(w.n_seasons)}시즌`;
}

/** 누적 표본: 타자 "3,012타석", 투수 "95.2이닝 · 상대 타자 402명" */
export function sampleText(rd: RoleDetail): string {
  if (rd.role === "bat") return `${fmtCount(rd.detail.sample_pa)}타석`;
  return `${fmtIp(rd.detail.sample_ip)}이닝 · 상대 타자 ${fmtCount(rd.detail.sample_bf)}명`;
}

/** 그룹 이름 (D13). 경력 그룹이 아니면 사용 시즌 수로 정한다 */
export function groupLabel(group: GroupName, w: InputWindow): string {
  if (group === "seasons_1_2" || group === "seasons_3_5") return CAREER_LABEL[group];
  return CAREER_LABEL[careerGroup(w.n_seasons)];
}

export const METRIC_LABEL: Record<Metric, string> = {
  ops: "OPS",
  pa: "PA",
  hr: "HR",
  era: "ERA",
  ip: "IP",
  k_bb_pct: "K−BB%",
};

/** 시즌별 추이 차트에 그리는 지표 (역할의 대표 비율 지표) */
export function chartMetric(role: Role): BandMetric {
  return role === "bat" ? "ops" : "era";
}

// ---------------------------------------------------------------------------
// 2027 예측 카드
// ---------------------------------------------------------------------------

export interface ProjectionCard {
  metric: Metric;
  /** "OPS" — 화면에는 "2027 OPS"로 */
  label: string;
  /** 출전량·HR 표시 (D30) */
  isStarred: boolean;
  value: string;
  /** "±.097" — 범위가 없으면 null (D29) */
  band: string | null;
  sub: string;
}

interface CardSpec {
  metric: Metric;
  value: number;
  prev: number | null | undefined;
  band?: number | null;
  league?: number;
  isStarred?: boolean;
  isPlayingTime?: boolean;
}

function card(spec: CardSpec, prevSeason: number | null, lastSeason: number): ProjectionCard {
  const parts: string[] = [];
  if (prevSeason === null) parts.push(`${lastSeason} 미출전`);
  else parts.push(`${prevSeason} 실제 ${fmtMetric(spec.metric, spec.prev)}`);
  if (spec.league !== undefined) {
    parts.push(`리그 평균 ${fmtMetric(spec.metric, spec.league)}보다 ${fmtDiff(spec.metric, spec.value - spec.league)}`);
  }
  if (spec.isPlayingTime === true) parts.push("보수적 예상");
  const band = spec.band == null ? null : fmtBand(spec.metric, spec.band);
  return {
    metric: spec.metric,
    label: METRIC_LABEL[spec.metric],
    isStarred: spec.isStarred === true,
    value: fmtMetric(spec.metric, spec.value),
    band: band === DASH ? null : band,
    sub: parts.join(" · "),
  };
}

/** 역할별 예측 카드 3개 (SA-2: 비율 지표 ±범위·리그 평균 대비, 출전량 *) */
export function projectionCards(rd: RoleDetail, targetSeason: number): ProjectionCard[] {
  const lastSeason = targetSeason - 1;
  if (rd.role === "bat") {
    const { projection: p, prev, league_baseline: league, group_error: ge } = rd.detail;
    const season = prev?.season ?? null;
    return [
      card({ metric: "ops", value: p.ops, prev: prev?.ops, band: ge.band80.ops, league: league.ops }, season, lastSeason),
      card({ metric: "pa", value: p.pa, prev: prev?.pa, isStarred: true, isPlayingTime: true }, season, lastSeason),
      card({ metric: "hr", value: p.hr, prev: prev?.hr, isStarred: true }, season, lastSeason),
    ];
  }
  const { projection: p, prev, league_baseline: league, group_error: ge } = rd.detail;
  const season = prev?.season ?? null;
  return [
    card({ metric: "era", value: p.era, prev: prev?.era, band: ge.band80.era, league: league.era }, season, lastSeason),
    card({ metric: "ip", value: p.ip, prev: prev?.ip, isStarred: true, isPlayingTime: true }, season, lastSeason),
    card(
      { metric: "k_bb_pct", value: p.k_bb_pct, prev: prev?.k_bb_pct, band: ge.band80.k_bb_pct, league: league.k_bb_pct },
      season,
      lastSeason,
    ),
  ];
}

// ---------------------------------------------------------------------------
// 그룹 과거 오차 문장 ("이 예측은 얼마나 빗나갈까")
// ---------------------------------------------------------------------------

function rateSentence(metric: BandMetric, band: number | null | undefined, mae: number | null | undefined): string {
  const label = METRIC_LABEL[metric];
  const hasBand = band != null && Number.isFinite(band);
  const hasMae = mae != null && Number.isFinite(mae);
  if (hasBand && hasMae) {
    return `${label}가 10명 중 8명꼴로 ${fmtBand(metric, band)} 안에 들어왔고, 평균 ${fmtMae(metric, mae)} 빗나갔습니다.`;
  }
  if (hasBand) return `${label}가 10명 중 8명꼴로 ${fmtBand(metric, band)} 안에 들어왔습니다.`;
  if (hasMae) return `${label}는 평균 ${fmtMae(metric, mae)} 빗나갔습니다.`;
  return `${label}는 견줄 과거 검증 자료가 없습니다.`;
}

interface CountPart {
  metric: Metric;
  /** 지표 이름 뒤 조사 ("PA는", "HR은") */
  particle: string;
  unit: string;
  mae: number | null | undefined;
}

/** 출전량·HR의 평균 오차 한 문장: "PA는 평균 131.0타석, HR은 평균 7.20개 빗나갔습니다." */
function countSentence(parts: CountPart[]): string {
  const known = parts.filter((p) => p.mae != null && Number.isFinite(p.mae));
  const missing = parts.filter((p) => !known.includes(p));
  const sentences: string[] = [];
  if (known.length > 0) {
    const body = known.map((p) => `${METRIC_LABEL[p.metric]}${p.particle} 평균 ${fmtMae(p.metric, p.mae)}${p.unit}`);
    sentences.push(`${body.join(", ")} 빗나갔습니다.`);
  }
  if (missing.length > 0) {
    sentences.push(`${missing.map((p) => METRIC_LABEL[p.metric]).join("·")}는 견줄 과거 검증 자료가 없습니다.`);
  }
  return sentences.join(" ");
}

/** 비슷한 경력 그룹의 과거 오차를 한 문단으로 (D13·D29) */
export function errorSummary(rd: RoleDetail): string {
  const { group_error: ge, window } = rd.detail;
  const head = `비슷한 경력(${groupLabel(ge.group, window)}) ${ROLE_NOUN[rd.role]}들의 과거 검증 결과입니다.`;
  if (rd.role === "bat") {
    return [
      head,
      rateSentence("ops", ge.band80.ops, ge.mae.ops),
      countSentence([
        { metric: "pa", particle: "는", unit: "타석", mae: ge.mae.pa },
        { metric: "hr", particle: "은", unit: "개", mae: ge.mae.hr },
      ]),
    ].join(" ");
  }
  return [
    head,
    rateSentence("era", ge.band80.era, ge.mae.era),
    rateSentence("k_bb_pct", ge.band80.k_bb_pct, ge.mae.k_bb_pct),
    countSentence([{ metric: "ip", particle: "는", unit: "이닝", mae: ge.mae.ip }]),
  ].join(" ");
}

// ---------------------------------------------------------------------------
// 수비 (D21) — format.ts에 OAA·FRV 전용 함수가 없어 같은 자릿수의 표기를 빌린다
// ---------------------------------------------------------------------------

/** OAA(아웃 단위 정수): "+6", "−2", "0" */
export function fmtOaa(v: number | null | undefined): string {
  if (v === 0) return fmtCount(0);
  return fmtDiff("hr", v);
}

/** FRV(실점 단위, 소수 1자리): "+4.9", "−1.4", "0.0" */
export function fmtFrv(v: number | null | undefined): string {
  if (v === 0) return fmtPct(0);
  return fmtDiff("frv", v);
}
