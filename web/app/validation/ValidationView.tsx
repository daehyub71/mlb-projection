"use client";

// 검증 화면 (SA-4, SPEC F15·D13·D29·D31) — validation.json을 읽고, WAR 구역(SA-8)은 war.json을 읽는다 (DESIGN §1·§4-1)
// 틀은 Astryx table-page 템플릿(Layout + LayoutContent padding 0 + VStack padding 4·gap 8)에서 왔다.
// 머리글이 sticky라 문서 전체가 스크롤되도록 height="auto"로 두고, 폭은 시안(SA-4)의 1200 열을 따른다.
import { useMemo, useState } from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { Card } from "@astryxdesign/core/Card";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { Grid } from "@astryxdesign/core/Grid";
import { Heading } from "@astryxdesign/core/Heading";
import { HStack } from "@astryxdesign/core/HStack";
import { Layout, LayoutContent } from "@astryxdesign/core/Layout";
import { Link } from "@astryxdesign/core/Link";
import { ProgressBar } from "@astryxdesign/core/ProgressBar";
import { Selector } from "@astryxdesign/core/Selector";
import { Spinner } from "@astryxdesign/core/Spinner";
import { Table, type TableColumn } from "@astryxdesign/core/Table";
import { Text } from "@astryxdesign/core/Text";
import { Token } from "@astryxdesign/core/Token";
import { VStack } from "@astryxdesign/core/VStack";
import { loadValidation, loadWar } from "@/src/data/load";
import type {
  BandMetric,
  BandRow,
  CareerGroup,
  EvalRow,
  FinalTest,
  GroupName,
  Metric,
  ModelName,
  Role,
  ValidationReport,
} from "@/src/data/types";
import { useJson } from "@/src/data/useJson";
import { DASH, fmtBand, fmtCount, fmtMae } from "@/src/format";
import { CAREER_LABEL, RERUN_STORY } from "@/src/site";
import { SiteFooter } from "../components/SiteFooter";
import { SiteHeader } from "../components/SiteHeader";
import { WarValidation } from "./WarValidation";

// ---------------------------------------------------------------------------
// 이름표 (계약 site.ts에 없어 이 화면에 둔다 — 시안 SA-4 문구)
// ---------------------------------------------------------------------------

const METRIC_LABEL: Record<Metric, string> = {
  ops: "OPS",
  pa: "PA",
  hr: "HR",
  era: "ERA",
  ip: "IP",
  k_bb_pct: "K−BB%",
};

const ROLE_LABEL: Record<Role, string> = { bat: "타자", pit: "투수" };

const MODEL_LABEL: Record<ModelName, string> = {
  extended: "확장",
  weighted: "가중 + 리그 평균 회귀",
  league: "리그 평균",
  baseline: "직전 시즌",
};

const GROUP_LABEL: Record<GroupName, string> = {
  all: "전체",
  seasons_1_2: CAREER_LABEL.seasons_1_2,
  seasons_3_5: CAREER_LABEL.seasons_3_5,
  prev_played: "직전 시즌 출전",
  prev_absent: "직전 시즌 미출전",
};

const GROUP_ORDER: GroupName[] = ["all", "seasons_1_2", "seasons_3_5", "prev_played", "prev_absent"];
const MODEL_ORDER: ModelName[] = ["extended", "weighted", "league", "baseline"];
const METRIC_ORDER: Record<Role, Metric[]> = { bat: ["ops", "pa", "hr"], pit: ["era", "ip", "k_bb_pct"] };

/** 2026 최종 검증 카드 6개 (시안 순서). 대상 설명은 SPEC D13 평가 집단 */
const CARD_METRICS: ReadonlyArray<{ metric: Metric; role: Role; population: string }> = [
  { metric: "ops", role: "bat", population: "200타석 이상" },
  { metric: "k_bb_pct", role: "pit", population: "50이닝 이상" },
  { metric: "era", role: "pit", population: "50이닝 이상" },
  { metric: "pa", role: "bat", population: "타자" },
  { metric: "ip", role: "pit", population: "투수" },
  { metric: "hr", role: "bat", population: "타자" },
];

/** ± 범위 표의 행 (비율 지표만, D29) */
const BAND_METRICS: ReadonlyArray<{ metric: BandMetric; role: Role }> = [
  { metric: "ops", role: "bat" },
  { metric: "era", role: "pit" },
  { metric: "k_bb_pct", role: "pit" },
];

const CAREER_GROUPS: CareerGroup[] = ["seasons_1_2", "seasons_3_5"];
const CAREER_PHRASE: Record<CareerGroup, string> = {
  seasons_1_2: `짧은 경력(${CAREER_LABEL.seasons_1_2})`,
  seasons_3_5: `긴 경력(${CAREER_LABEL.seasons_3_5})`,
};

const COUNT_WORD: Record<number, string> = { 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯" };
const countWord = (n: number) => COUNT_WORD[n] ?? fmtCount(n);

// ---------------------------------------------------------------------------
// 순수 계산
// ---------------------------------------------------------------------------

/** 그 역할·시즌에 확장 모델 행이 하나라도 있으면 확장이 최종 모델, 없으면 가중 모델 (최종 모델 규칙, PLAN) */
function finalModelOf(rows: EvalRow[], role: Role, season: number): ModelName {
  return rows.some((r) => r.role === role && r.target_season === season && r.model === "extended")
    ? "extended"
    : "weighted";
}

function findRow(
  rows: EvalRow[],
  season: number,
  role: Role,
  model: ModelName,
  metric: Metric,
  group: GroupName,
): EvalRow | undefined {
  return rows.find(
    (r) =>
      r.target_season === season &&
      r.role === role &&
      r.model === model &&
      r.metric === metric &&
      r.group === group,
  );
}

interface Comparison {
  metric: Metric;
  mae: number;
  baseName: "리그 평균" | "직전 시즌";
  base: number;
}

/** 최종 모델 행이 기준선(리그 평균·직전 시즌)보다 오차가 작지 않은 비교들 */
function lostComparisons(row: EvalRow): Comparison[] {
  if (row.mae == null) return [];
  const out: Comparison[] = [];
  const bases: Array<[Comparison["baseName"], number | null]> = [
    ["리그 평균", row.baseline_mae_league],
    ["직전 시즌", row.baseline_mae_prev],
  ];
  for (const [baseName, base] of bases) {
    if (base != null && !(row.mae < base)) out.push({ metric: row.metric, mae: row.mae, baseName, base });
  }
  return out;
}

/** 결과 요약 문장 — 전체 그룹 승패, 진 경력 그룹(숫자 포함), MAE 읽는 법 */
function summaryText(rows: EvalRow[], season: number): string {
  const sentences: string[] = [];

  // 1) 전체 그룹
  const allRows = CARD_METRICS.map(({ metric, role }) =>
    findRow(rows, season, role, finalModelOf(rows, role, season), metric, "all"),
  ).filter((r): r is EvalRow => r !== undefined && r.mae != null);
  const allLosses = allRows.flatMap(lostComparisons);
  if (allLosses.length === 0 && allRows.length > 0) {
    sentences.push(`${countWord(allRows.length)} 지표 모두 리그 평균과 직전 시즌보다 오차가 작았습니다.`);
  } else if (allLosses.length > 0) {
    const list = allLosses
      .map((c) => `${METRIC_LABEL[c.metric]} ${fmtMae(c.metric, c.mae)} 대 ${c.baseName} ${fmtMae(c.metric, c.base)}`)
      .join(", ");
    const lostMetrics = new Set(allLosses.map((c) => c.metric));
    sentences.push(`전체 선수로 보면 기준선보다 오차가 작지 않은 지표가 있었습니다 — ${list}.`);
    if (lostMetrics.size < allRows.length) sentences.push("나머지 지표는 리그 평균과 직전 시즌보다 오차가 작았습니다.");
  }

  // 2) 경력 그룹 (1~2시즌·3~5시즌)
  const groupLosses: Array<{ group: CareerGroup; items: Array<{ metric: Metric; losses: Comparison[] }> }> = [];
  const baseNames = new Set<Comparison["baseName"]>();
  let groupRowCount = 0;
  for (const group of CAREER_GROUPS) {
    const items: Array<{ metric: Metric; losses: Comparison[] }> = [];
    for (const { metric, role } of CARD_METRICS) {
      const row = findRow(rows, season, role, finalModelOf(rows, role, season), metric, group);
      if (row === undefined || row.mae == null) continue;
      groupRowCount += 1;
      const losses = lostComparisons(row);
      if (losses.length === 0) continue;
      losses.forEach((c) => baseNames.add(c.baseName));
      items.push({ metric, losses });
    }
    if (items.length > 0) groupLosses.push({ group, items });
  }
  // 기준선이 하나뿐이면 괄호 안에서 이름을 빼고 문장 끝에 한 번 적는다 (시안 문장 꼴)
  const single = baseNames.size === 1 ? [...baseNames][0]! : null;
  const phrases = groupLosses.map(({ group, items }) => {
    const metrics = items.map(({ metric, losses }) => {
      const vs = losses
        .map((c) => (single === null ? `${c.baseName} ${fmtMae(metric, c.base)}` : fmtMae(metric, c.base)))
        .join(" · ");
      return `${METRIC_LABEL[metric]}(${fmtMae(metric, losses[0]!.mae)} 대 ${vs})`;
    });
    return `${CAREER_PHRASE[group]} 선수의 ${metrics.join(", ")}`;
  });
  if (phrases.length > 0) {
    const against =
      single === "직전 시즌" ? "직전 시즌을 그대로 쓴 값" : single === "리그 평균" ? "리그 평균을 그대로 찍은 값" : "기준선";
    sentences.push(`그룹으로 나누면 ${phrases.join(", ")}만 ${against}보다 오차가 컸습니다.`);
  } else if (groupRowCount > 0) {
    sentences.push(
      `경력 그룹(${CAREER_LABEL.seasons_1_2}·${CAREER_LABEL.seasons_3_5})으로 나눠도 모든 지표가 두 기준선보다 오차가 작았습니다.`,
    );
  }

  sentences.push("수치는 평균 절대 오차(MAE)이고 낮을수록 좋습니다.");
  return sentences.join(" ");
}

/** 다시 돌렸을 때 첫 결과 한 줄 (D31) — 지금과 표기가 다른 지표만 숫자로, 같은 지표는 묶어 적는다 */
function firstRunText(rows: EvalRow[], ft: FinalTest): string | null {
  if (ft.first_run === null) return null;
  const changed: string[] = [];
  const same: string[] = [];
  for (const { metric, role } of CARD_METRICS) {
    const first = ft.first_run.mae[metric];
    if (first == null) continue;
    const now = findRow(rows, ft.target_season, role, finalModelOf(rows, role, ft.target_season), metric, "all");
    if (now !== undefined && fmtMae(metric, now.mae) === fmtMae(metric, first)) same.push(METRIC_LABEL[metric]);
    else changed.push(`${METRIC_LABEL[metric]} ${fmtMae(metric, first)}`);
  }
  const which = ft.reruns.length === 1 ? "두 번째" : "마지막";
  if (changed.length === 0) return `첫 결과도 지금과 같았습니다 → 위 카드는 ${which} 결과입니다.`;
  const sameNote = same.length > 0 ? ` (${same.join("·")} 값은 같음)` : "";
  return `첫 결과: ${changed.join(" · ")}${sameNote} → 위 카드는 ${which} 결과입니다.`;
}

// ---------------------------------------------------------------------------
// 화면 조각
// ---------------------------------------------------------------------------

function MetricCard({ row, metric, population }: { row: EvalRow; metric: Metric; population: string }) {
  // 막대 이름표는 값보다 앞에, 이 모델 → 리그 평균(비율 지표만) → 직전 시즌 순서. 값은 글자로도 보인다 (N10)
  const bars: Array<{ label: string; value: number | null; variant: "accent" | "neutral" }> = [
    { label: "이 모델", value: row.mae, variant: "accent" },
  ];
  if (row.baseline_mae_league != null) bars.push({ label: "리그 평균", value: row.baseline_mae_league, variant: "neutral" });
  if (row.baseline_mae_prev != null) bars.push({ label: "직전 시즌", value: row.baseline_mae_prev, variant: "neutral" });
  const max = Math.max(...bars.map((b) => b.value ?? 0));

  return (
    <Card>
      <VStack gap={3}>
        <HStack gap={2} hAlign="between" vAlign="center" wrap="wrap">
          <Heading level={3}>{METRIC_LABEL[metric]}</Heading>
          <Text type="supporting" color="secondary" hasTabularNumbers>
            {`${population} ${fmtCount(row.n)}명`}
          </Text>
        </HStack>
        <VStack gap={2}>
          {bars.map((bar) => (
            <ProgressBar
              key={bar.label}
              label={bar.label}
              value={bar.value ?? 0}
              max={max > 0 ? max : 1}
              hasValueLabel
              formatValueLabel={() => fmtMae(metric, bar.value)}
              variant={bar.variant}
            />
          ))}
        </VStack>
      </VStack>
    </Card>
  );
}

function FinalTestSection({ report, holdout }: { report: ValidationReport; holdout: number }) {
  const ft = report.final_test;
  if (ft === null) {
    return (
      <VStack as="section" gap={4} aria-labelledby="final-test">
        <Heading id="final-test" level={2}>
          {`${holdout} 최종 검증`}
        </Heading>
        <Text as="p" color="secondary">
          {`${holdout} 최종 검증은 아직 돌리지 않았습니다. 아래 표는 설정을 정할 때 쓴 시즌의 결과입니다.`}
        </Text>
      </VStack>
    );
  }

  const season = ft.target_season;
  const cards = CARD_METRICS.map(({ metric, role, population }) => ({
    metric,
    population,
    row: findRow(report.rows, season, role, finalModelOf(report.rows, role, season), metric, "all"),
  })).filter((c): c is { metric: Metric; population: string; row: EvalRow } => c.row !== undefined);
  const reruns = ft.reruns.length;
  const firstRun = reruns > 0 ? firstRunText(report.rows, ft) : null;

  return (
    <VStack as="section" gap={4} aria-labelledby="final-test">
      <HStack gap={3} hAlign="between" vAlign="center" wrap="wrap">
        <Heading id="final-test" level={2}>
          {`${season} 최종 검증`}
        </Heading>
        <Token
          label={
            reruns > 0
              ? `고정한 설정으로 · 다시 돌림 ${fmtCount(reruns)}번 (아래 이유)`
              : "고정한 설정으로 · 한 번만 돌림"
          }
        />
      </HStack>
      {cards.length === 0 ? (
        <EmptyState title={`${season} 최종 검증 행이 자료에 없습니다`} isCompact />
      ) : (
        <>
          <Grid columns={{ minWidth: 300, max: 3 }} gap={4}>
            {cards.map((c) => (
              <MetricCard key={c.metric} row={c.row} metric={c.metric} population={c.population} />
            ))}
          </Grid>
          <Text as="p">{summaryText(report.rows, season)}</Text>
        </>
      )}
      {reruns > 0 && (
        <Card variant="muted">
          <VStack gap={2}>
            <Heading level={3}>{`${season} 시험을 ${countWord(reruns + 1)} 번 돌린 이유`}</Heading>
            {/* 사람이 읽는 설명(SA-4 합의 문구) 뒤에 기록 파일의 사유를 그대로 둔다 — 숨기지 않는다 */}
            {reruns === 1 && firstRun !== null && <Text as="p">{RERUN_STORY}</Text>}
            {ft.reruns.map((r) => (
              <Text key={r.ran_at} as="p" type="supporting" color="secondary">
                {`기록된 사유: ${r.reason}`}
              </Text>
            ))}
            {firstRun !== null && (
              <Text as="p" hasTabularNumbers>
                {firstRun}
              </Text>
            )}
          </VStack>
        </Card>
      )}
    </VStack>
  );
}

type BandTableRow = {
  id: string;
  metric: BandMetric;
  role: Role;
  short: BandRow | undefined;
  long: BandRow | undefined;
};

const bandCount = (b: BandRow | undefined) => (b === undefined ? DASH : `${fmtCount(b.n)}명`);

const BAND_COLUMNS: TableColumn<BandTableRow>[] = [
  {
    key: "metric",
    header: "지표",
    renderCell: (r) => <Text weight="bold">{METRIC_LABEL[r.metric]}</Text>,
  },
  {
    key: "short",
    header: `${CAREER_LABEL.seasons_1_2} 경력`,
    align: "end",
    renderCell: (r) => (
      <Text size="lg" weight="bold" hasTabularNumbers>
        {fmtBand(r.metric, r.short?.q80)}
      </Text>
    ),
  },
  {
    key: "long",
    header: `${CAREER_LABEL.seasons_3_5} 경력`,
    align: "end",
    renderCell: (r) => (
      <Text size="lg" weight="bold" hasTabularNumbers>
        {fmtBand(r.metric, r.long?.q80)}
      </Text>
    ),
  },
  {
    key: "who",
    header: "대상",
    renderCell: (r) => (
      <Text color="secondary" hasTabularNumbers>
        {`${ROLE_LABEL[r.role]} ${bandCount(r.short)} · ${bandCount(r.long)}`}
      </Text>
    ),
  },
];

function BandSection({ bands }: { bands: BandRow[] }) {
  const data: BandTableRow[] = BAND_METRICS.map(({ metric, role }) => {
    const pick = (group: CareerGroup) => bands.find((b) => b.role === role && b.metric === metric && b.group === group);
    return { id: `${role}-${metric}`, metric, role, short: pick("seasons_1_2"), long: pick("seasons_3_5") };
  }).filter((r) => r.short !== undefined || r.long !== undefined);
  const targets = [...new Set(bands.flatMap((b) => b.targets))].sort((a, b) => a - b);
  const span =
    targets.length === 0 ? "과거" : targets.length === 1 ? `${targets[0]}` : `${targets[0]}~${targets[targets.length - 1]}`;

  return (
    <VStack as="section" gap={4} aria-labelledby="bands">
      <Heading id="bands" level={2}>
        ± 범위는 어떻게 정했나
      </Heading>
      <Text as="p" color="secondary">
        {`예측표의 ± 값은 ${span} 시즌의 실제 오차에서 나왔습니다. 비슷한 경력의 선수 10명 중 8명은 실제 성적이 예측 ± 이 범위 안에 들어왔습니다. 출전량(PA·IP)과 HR은 0에 가까운 예측이 섞여 범위가 고르지 않아 평균 오차만 보여 줍니다.`}
      </Text>
      <Table<BandTableRow>
        data={data}
        columns={BAND_COLUMNS}
        idKey="id"
        aria-labelledby="bands"
        emptyState={<EmptyState title="± 범위 자료가 없습니다" isCompact />}
      />
    </VStack>
  );
}

type DetailRow = {
  id: string;
  model: ModelName;
  metric: Metric;
  n: number;
  mae: number | null;
  isFinal: boolean;
};

const DETAIL_COLUMNS: TableColumn<DetailRow>[] = [
  {
    key: "model",
    header: "모델",
    renderCell: (r) =>
      r.isFinal ? (
        <Text weight="bold" color="accent">{`${MODEL_LABEL[r.model]} (최종)`}</Text>
      ) : (
        <Text>{MODEL_LABEL[r.model]}</Text>
      ),
  },
  { key: "metric", header: "지표", renderCell: (r) => <Text>{METRIC_LABEL[r.metric]}</Text> },
  {
    key: "n",
    header: "대상 수",
    align: "end",
    renderCell: (r) => (
      <Text color="secondary" hasTabularNumbers>
        {fmtCount(r.n)}
      </Text>
    ),
  },
  {
    key: "mae",
    header: "평균 오차",
    align: "end",
    renderCell: (r) => (
      <Text weight={r.isFinal ? "bold" : "normal"} hasTabularNumbers>
        {fmtMae(r.metric, r.mae)}
      </Text>
    ),
  },
];

const ROLE_OPTIONS = (["bat", "pit"] as const).map((r) => ({ value: r, label: ROLE_LABEL[r] }));
const GROUP_OPTIONS = GROUP_ORDER.map((g) => ({ value: g, label: GROUP_LABEL[g] }));

function DetailSection({ rows }: { rows: EvalRow[] }) {
  const seasons = useMemo(() => [...new Set(rows.map((r) => r.target_season))].sort((a, b) => b - a), [rows]);
  const [pickedSeason, setPickedSeason] = useState<string | null>(null);
  const [role, setRole] = useState<Role>("bat");
  const [group, setGroup] = useState<GroupName>("all");
  // 기본은 최신 목표 시즌
  const season = pickedSeason ?? (seasons.length > 0 ? String(seasons[0]) : null);

  const data: DetailRow[] = useMemo(() => {
    if (season === null) return [];
    const target = Number(season);
    const finalModel = finalModelOf(rows, role, target);
    return rows
      .filter((r) => r.target_season === target && r.role === role && r.group === group)
      .sort(
        (a, b) =>
          METRIC_ORDER[role].indexOf(a.metric) - METRIC_ORDER[role].indexOf(b.metric) ||
          MODEL_ORDER.indexOf(a.model) - MODEL_ORDER.indexOf(b.model),
      )
      .map((r) => ({
        id: `${r.model}-${r.metric}`,
        model: r.model,
        metric: r.metric,
        n: r.n,
        mae: r.mae,
        isFinal: r.model === finalModel,
      }));
  }, [rows, season, role, group]);

  return (
    <VStack as="section" gap={4} aria-labelledby="detail">
      <Heading id="detail" level={2}>
        모델별·지표별 오차
      </Heading>
      <HStack gap={4} wrap="wrap" vAlign="end">
        <Selector
          label="목표 시즌"
          options={seasons.map((s) => String(s))}
          value={season ?? undefined}
          onChange={(v) => setPickedSeason(v)}
        />
        <Selector label="역할" options={ROLE_OPTIONS} value={role} onChange={(v) => setRole(v === "pit" ? "pit" : "bat")} />
        <Selector
          label="그룹"
          options={GROUP_OPTIONS}
          value={group}
          onChange={(v) => setGroup(GROUP_ORDER.find((g) => g === v) ?? "all")}
        />
      </HStack>
      <Table<DetailRow>
        data={data}
        columns={DETAIL_COLUMNS}
        idKey="id"
        aria-labelledby="detail"
        emptyState={<EmptyState title="이 조합의 검증 결과가 없습니다" isCompact />}
      />
      <Text as="p" type="supporting" color="secondary">
        OPS·ERA·K−BB%는 그해 200타석·50이닝 이상 선수, PA·IP·HR은 예측 대상 전원(그해 뛰지 않았으면 0)으로 셉니다.{" "}
        <Link href="/about/">방법 설명</Link>
      </Text>
    </VStack>
  );
}

// ---------------------------------------------------------------------------
// 화면
// ---------------------------------------------------------------------------

export function ValidationView() {
  const state = useJson(() => loadValidation(), "validation");
  // WAR 구역(SA-8)은 war.json을 따로 받는다 — validation.json과 나란히, 실패해도 기존 구역은 그대로
  const warState = useJson(() => loadWar(), "war");
  const ranFinal = state.status !== "ok" || state.data.final_test !== null;

  return (
    <VStack gap={0} minHeight="100dvh">
      <SiteHeader active="validation" />
      <Layout
        height="auto"
        contentWidth={1200}
        content={
          <LayoutContent role="main" padding={0} isScrollable={false}>
            <VStack padding={4} gap={8} paddingBlock={8}>
              <VStack gap={3}>
                <Text type="label" color="accent" weight="bold">
                  BACKTEST · HOW CLOSE WERE WE
                </Text>
                <Heading level={1} type="display-2">
                  예측은 얼마나 맞았나
                </Heading>
                <Text as="p" type="large" color="secondary">
                  {ranFinal
                    ? "같은 방법으로 지난 시즌을 미리 맞혀 보고, 실제 성적과의 평균 오차를 그대로 공개합니다. 설정은 2024·2025 시즌만으로 정했고, 그 설정을 고정한 채 2026 시즌을 시험했습니다. 리그 평균을 그대로 찍은 값, 직전 시즌을 그대로 쓴 값과 나란히 봅니다."
                    : "같은 방법으로 지난 시즌을 미리 맞혀 보고, 실제 성적과의 평균 오차를 그대로 공개합니다. 설정은 2024·2025 시즌만으로 정하고, 그 설정을 고정한 채 2026 시즌을 시험합니다. 리그 평균을 그대로 찍은 값, 직전 시즌을 그대로 쓴 값과 나란히 봅니다."}
                </Text>
              </VStack>

              {state.status === "loading" && (
                <HStack hAlign="center" paddingBlock={8}>
                  <Spinner size="lg" label="검증 결과를 불러오는 중…" aria-label="검증 결과를 불러오는 중" />
                </HStack>
              )}

              {state.status === "error" && (
                <Banner
                  status="error"
                  title="검증 결과를 불러오지 못했습니다"
                  description={
                    state.notFound
                      ? "검증 자료 파일이 아직 없습니다. 잠시 뒤 다시 열어 주세요."
                      : "잠시 뒤 다시 열어 주세요. 문제가 계속되면 깃허브 저장소에 알려 주세요."
                  }
                />
              )}

              {state.status === "ok" && (
                <>
                  <FinalTestSection
                    report={state.data}
                    holdout={state.data.final_test?.target_season ?? Math.max(...state.data.meta.input_seasons)}
                  />
                  <BandSection bands={state.data.bands} />
                  <DetailSection rows={state.data.rows} />
                  <WarValidation state={warState} />
                </>
              )}
            </VStack>
          </LayoutContent>
        }
      />
      <SiteFooter />
    </VStack>
  );
}
