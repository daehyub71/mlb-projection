// 화면 숫자 표기 — 모든 화면이 이 함수만 쓴다 (DESIGN §4 v0.2 표시 규칙)
import type { Metric } from "./data/types";

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
