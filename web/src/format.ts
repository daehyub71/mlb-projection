// 화면 숫자 표기 — 모든 화면이 이 함수만 쓴다 (DESIGN §4 v0.2 표시 규칙)
import type { Metric, WarRoles } from "./data/types";

export const DASH = "—";
const MINUS = "−"; // U+2212

/** OPS·OBP·SLG·wOBA: 소수 3자리, 1 미만이면 앞의 0을 뺀다 (.939, 1.033) */
export function fmtRate(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return DASH;
  const sign = v < 0 ? MINUS : "";
  const text = Math.abs(v).toFixed(3);
  return sign + (text.startsWith("0.") ? text.slice(1) : text);
}

/** ERA·FIP·xERA: 소수 2자리 */
export function fmtEra(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? DASH : v.toFixed(2);
}

/** 백분율 지표(K%·BB%·K−BB%·Barrel% 등): 소수 1자리 */
export function fmtPct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return DASH;
  return (v < 0 ? MINUS : "") + Math.abs(v).toFixed(1);
}

/** PA·HR·G 같은 개수: 정수로 반올림, 천 단위 쉼표 */
export function fmtCount(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? DASH : Math.round(v).toLocaleString("ko-KR");
}

/** 소수 이닝(아웃/3) → 야구 표기 76.7 → "76.2" (아웃 3개 = 1이닝) */
export function fmtIp(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return DASH;
  const outs = Math.round(v * 3);
  return `${Math.floor(outs / 3).toLocaleString("ko-KR")}.${outs % 3}`;
}

/** 지표 이름에 맞는 예측·기록 표기 */
export function fmtMetric(metric: Metric | string, v: number | null | undefined): string {
  switch (metric) {
    case "ops":
    case "obp":
    case "slg":
    case "woba":
    case "xwoba":
    case "xwoba_against":
      return fmtRate(v);
    case "era":
    case "fip":
    case "xera":
      return fmtEra(v);
    case "pa":
    case "hr":
    case "g":
    case "gs":
      return fmtCount(v);
    case "ip":
      return fmtIp(v);
    default:
      return fmtPct(v);
  }
}

/** 과거 오차 80% 범위: "±.096", "±1.38", "±5.5" (SPEC D29) */
export function fmtBand(metric: Metric | string, v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return DASH;
  return `±${fmtMetric(metric, Math.abs(v))}`;
}

/** 리그 평균 대비 차이: "+.179", "−.019", "+0.35" */
export function fmtDiff(metric: Metric | string, v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return DASH;
  const body = fmtMetric(metric, Math.abs(v));
  return (v < 0 ? MINUS : "+") + body;
}

/** 평균 절대 오차(MAE): OPS .0615 · ERA .843 · K−BB% 3.62 · PA/IP 72.4 · HR 2.61 */
export function fmtMae(metric: Metric | string, v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return DASH;
  const digits: Record<string, number> = { ops: 4, era: 3, k_bb_pct: 2, pa: 1, ip: 1, hr: 2 };
  const text = v.toFixed(digits[metric] ?? 3);
  return text.startsWith("0.") && (metric === "ops" || metric === "era") ? text.slice(1) : text;
}

// ---------------------------------------------------------------------------
// v0.5 WAR (DESIGN §4-1 WAR 표시 규칙)
// ---------------------------------------------------------------------------

/** 부호를 붙여 소수 1자리 — 반올림해 0이면 부호 없이 "0.0" */
function signed1(v: number, plus: string): string {
  const text = Math.abs(v).toFixed(1);
  if (Number(text) === 0) return "0.0";
  return (v < 0 ? MINUS : plus) + text;
}

/** WAR: 소수 1자리 (6.18 → "6.2", −0.35 → "−0.3"). JSON은 2자리 */
export function fmtWar(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? DASH : signed1(v, "");
}

/** WAR 소수 2자리 — 시즌 표 · 구성 항목 합계 ("5.59", "3.69", "−0.25") */
export function fmtWar2(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return DASH;
  const text = Math.abs(v).toFixed(2);
  return Number(text) === 0 ? "0.00" : (v < 0 ? MINUS : "") + text;
}

/** 투타 겸업 분해 "타 3.9 + 투 2.3" — 두 역할이 다 있을 때만, 아니면 "" (D34) */
export function fmtWarSplit(bat: number | null | undefined, pit: number | null | undefined): string {
  if (bat == null || pit == null || !Number.isFinite(bat) || !Number.isFinite(pit)) return "";
  return `타 ${fmtWar(bat)} + 투 ${fmtWar(pit)}`;
}

/** 타자 WAR 구성 항목(점): "+29.2", "−14.5", "0.0" (E29) */
export function fmtRuns(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? DASH : signed1(v, "+");
}

/** WAR 평균 오차(승): 1 미만은 ".457", 1 이상은 "1.48" (DESIGN SA-8) */
export function fmtWarMae(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return DASH;
  return v < 1 ? v.toFixed(3).replace(/^0\./, ".") : v.toFixed(2);
}

const WAR_ROLE_LABEL: Record<WarRoles, string> = { bat: "타", pit: "투", both: "투타" };

/** 역할 배지 글자: 타 / 투 / 투타 */
export function warRoleLabel(roles: WarRoles): string {
  return WAR_ROLE_LABEL[roles];
}
