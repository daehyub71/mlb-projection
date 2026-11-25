"use client";

// 시즌별 추이 차트 (SA-2, SPEC D26) — 실제 값 선 + 2027 예측 점과 ±범위 막대 + 리그 평균선, 2026→2027 점선
import { useMemo } from "react";
import {
  Chart,
  ChartAxis,
  ChartGrid,
  dot,
  errorBar,
  line,
  referenceLine,
  useChart,
  xPixel,
  type TooltipSeriesValue,
} from "@astryxdesign/charts";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import type { BandMetric } from "@/src/data/types";
import { fmtBand, fmtMetric } from "@/src/format";
import { METRIC_LABEL } from "./text";

// 데이터 키는 차트가 숨겨 두는 자료 표의 열 이름이 된다 — 한국어로 둔다
const X_KEY = "시즌";
const ACTUAL = "실제";
const PROJ = "예측";
const HIGH = "범위 위";
const LOW = "범위 아래";

// 색은 테마 토큰 (DESIGN §3: 실제는 자홍 채움, 예측은 밝은 자홍 글자색)
const ACTUAL_COLOR = "var(--color-accent)";
const PROJ_COLOR = "var(--color-text-accent)";
const LEAGUE_COLOR = "var(--color-text-secondary)";

export interface TrendPoint {
  season: number;
  value: number | null;
}

interface TrendChartProps {
  metric: BandMetric;
  /** 표에 보이는 시즌들 (오래된 순) */
  points: TrendPoint[];
  targetSeason: number;
  projection: number;
  band: number | null | undefined;
  league: number | undefined;
}

type Row = Record<string, string | number | null>;

function finite(v: number | null | undefined): v is number {
  return v != null && Number.isFinite(v);
}

/**
 * 시즌 사이 빈 해도 칸을 두어(선이 끊긴다) 축이 실제 시간 간격을 지키게 한다.
 * 마지막 기록 뒤 예측 직전 해(2026)까지도 빈 칸을 둔다 — 2026 미출전 선수의 2025와 2027 예측이 붙어 보이지 않게.
 */
function buildRows(
  points: TrendPoint[],
  targetSeason: number,
  projection: number,
  band: number | null | undefined,
  cap: number | null,
) {
  const bySeason = new Map(points.map((p) => [p.season, p.value]));
  const seasons = points.map((p) => p.season);
  const rows: Row[] = [];
  // 축 상한을 넘는 실제 값 — 칸 이름 → 실제 값 (점은 상한에 그리고 실제 값을 적는다)
  const clipped = new Map<string, number>();
  if (seasons.length > 0) {
    for (let s = Math.min(...seasons); s <= Math.max(targetSeason - 1, ...seasons); s += 1) {
      const v = bySeason.get(s);
      let plotted: number | null = finite(v) ? v : null;
      if (plotted !== null && cap !== null && plotted > cap) {
        clipped.set(String(s), plotted);
        plotted = cap;
      }
      rows.push({ [X_KEY]: String(s), [ACTUAL]: plotted, [PROJ]: null, [HIGH]: null, [LOW]: null });
    }
  }
  const targetLabel = `${targetSeason} 예측`;
  rows.push({
    [X_KEY]: targetLabel,
    [ACTUAL]: null,
    [PROJ]: projection,
    [HIGH]: finite(band) ? projection + band : null,
    [LOW]: finite(band) ? projection - band : null,
  });
  return { rows, targetLabel, clipped };
}

function yDomain(rows: Row[], league: number | undefined): [number, number] {
  const values = rows
    .flatMap((r) => [r[ACTUAL], r[PROJ], r[HIGH], r[LOW]])
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (finite(league)) values.push(league);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = max > min ? (max - min) * 0.12 : Math.max(Math.abs(max) * 0.1, 0.05);
  return [Math.max(0, min - pad), max + pad];
}

/** 화면 낭독기와 화면 모두에 읽어 주는 값 설명 (차트 subtitle → svg <desc>) */
function describe(
  metric: BandMetric,
  points: TrendPoint[],
  targetSeason: number,
  projection: number,
  band: number | null | undefined,
  league: number | undefined,
): string {
  const seasons = points.map((p) => `${p.season} ${fmtMetric(metric, p.value)}`).join(" · ");
  let text = `${targetSeason} 예측 ${fmtMetric(metric, projection)}`;
  if (finite(band)) {
    const low = fmtMetric(metric, projection - band);
    const high = fmtMetric(metric, projection + band);
    text += ` (과거 오차 80% 범위 ${fmtBand(metric, band)}: ${low}~${high})`;
  }
  if (seasons !== "") text = `${seasons} → ${text}`;
  if (finite(league)) text += ` · 리그 평균 ${fmtMetric(metric, league)}`;
  return text;
}

/** ERA 세로축 상한 = 리그 평균 × 이 값 (2026-11-25 사용자 결정 — 0.1이닝 ERA 108 같은 시즌이 축을 늘리지 않게) */
export const ERA_AXIS_FACTOR = 3;

/** 축 상한에 걸린 시즌 위에 실제 값을 적는다 ("↑ 108.00") */
function ClippedMarks({ clipped, metric }: { clipped: Map<string, number>; metric: BandMetric }) {
  const { data, xKey, xScale, yScale } = useChart();
  return (
    <>
      {data
        .filter((d) => clipped.has(String(d[xKey])))
        .map((d) => (
          <text
            key={String(d[xKey])}
            x={xPixel(d, xKey, xScale)}
            y={yScale(d[ACTUAL] as number) - 10}
            textAnchor="middle"
            fill={ACTUAL_COLOR}
            fontSize={12}
            pointerEvents="none"
          >
            {`↑ ${fmtMetric(metric, clipped.get(String(d[xKey])) ?? null)}`}
          </text>
        ))}
    </>
  );
}

/** 마지막 실제 시즌에서 2027 예측 점까지 점선 (DESIGN §4 v0.2 규칙) */
function ProjectionLink({ fromLabel, toLabel }: { fromLabel: string | null; toLabel: string }) {
  const { data, xKey, xScale, yScale } = useChart();
  if (fromLabel === null) return null;
  const from = data.find((d) => d[xKey] === fromLabel);
  const to = data.find((d) => d[xKey] === toLabel);
  if (from === undefined || to === undefined) return null;
  const fromValue = from[ACTUAL];
  const toValue = to[PROJ];
  if (typeof fromValue !== "number" || typeof toValue !== "number") return null;
  return (
    <line
      x1={xPixel(from, xKey, xScale)}
      y1={yScale(fromValue)}
      x2={xPixel(to, xKey, xScale)}
      y2={yScale(toValue)}
      stroke={PROJ_COLOR}
      strokeWidth={2}
      strokeDasharray="4 5"
      pointerEvents="none"
      aria-hidden="true"
    />
  );
}

export function TrendChart({ metric, points, targetSeason, projection, band, league }: TrendChartProps) {
  const label = METRIC_LABEL[metric];
  const cap = metric === "era" && finite(league) ? league * ERA_AXIS_FACTOR : null;
  const { rows, targetLabel, clipped } = useMemo(
    () => buildRows(points, targetSeason, projection, band, cap),
    [points, targetSeason, projection, band, cap],
  );
  // 마크 정의는 차트마다 새로 만든다 (Astryx: 한 정의를 여러 Chart에 다시 쓰지 않는다)
  const series = useMemo(
    () => [
      line(ACTUAL, { label: `실제 ${label}`, dots: true, strokeWidth: 3, color: ACTUAL_COLOR }),
      ...(finite(band) ? [errorBar({ high: HIGH, low: LOW, color: PROJ_COLOR, strokeWidth: 2, capWidth: 16 })] : []),
      dot(PROJ, { label: `${targetSeason} 예측`, radius: 6, opacity: 1, color: PROJ_COLOR }),
      ...(finite(league)
        ? [referenceLine({ y: league, label: `리그 평균 ${fmtMetric(metric, league)}`, color: LEAGUE_COLOR, labelPosition: "start" as const })]
        : []),
    ],
    [label, band, targetSeason, league, metric],
  );
  const domain = useMemo(() => yDomain(rows, league), [rows, league]);
  const lastActual = [...rows].reverse().find((r) => typeof r[ACTUAL] === "number");
  const fromLabel = lastActual === undefined ? null : String(lastActual[X_KEY]);
  const first = points[0]?.season;
  const last = points[points.length - 1]?.season;
  const span = first === undefined ? "" : first === last ? `${first} ` : `${first}~${last} `;

  const renderTooltip = (xValue: unknown, values: TooltipSeriesValue[]) => {
    const known = values.filter((v) => typeof v.value === "number");
    return (
      <VStack gap={1}>
        <Text weight="semibold">{String(xValue)}</Text>
        {known.length === 0 ? (
          // 빈 해(미출전·기록 없음) 칸
          <Text type="supporting" color="secondary">
            기록 없음
          </Text>
        ) : (
          known.map((v) => (
            <Text key={v.key} type="supporting" hasTabularNumbers>
              {v.label} {fmtMetric(metric, v.key === ACTUAL ? (clipped.get(String(xValue)) ?? (v.value as number)) : (v.value as number))}
              {v.key === PROJ && finite(band) ? ` ${fmtBand(metric, band)}` : ""}
            </Text>
          ))
        )}
      </VStack>
    );
  };

  return (
    <Chart
      data={rows}
      xKey={X_KEY}
      series={series}
      height={300}
      yDomain={domain}
      margin={{ left: 56 }}
      title={`${span}실제 ${label}와 ${targetLabel}`}
      subtitle={describe(metric, points, targetSeason, projection, band, league)}
      grid={<ChartGrid horizontal />}
      axes={
        <>
          <ChartAxis position="bottom" />
          <ChartAxis position="left" tickFormat={(v) => fmtMetric(metric, Number(v))} />
        </>
      }
      legend
      tooltip={{ render: renderTooltip }}
    >
      <ProjectionLink fromLabel={fromLabel} toLabel={targetLabel} />
      {clipped.size > 0 && <ClippedMarks clipped={clipped} metric={metric} />}
    </Chart>
  );
}
