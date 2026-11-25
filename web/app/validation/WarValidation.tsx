"use client";

// 검증 화면 WAR 구역 (SA-8, SPEC F22·D36·D39·D40·D41·E28·E29) — war.json(loadWar)을 받은 상태를 받아 그린다.
// 받기는 ValidationView가 validation.json과 나란히 한다. 기존 여섯 지표 구역 아래에 붙는다.
import { useMemo, useState } from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { Card } from "@astryxdesign/core/Card";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { Grid } from "@astryxdesign/core/Grid";
import { Heading } from "@astryxdesign/core/Heading";
import { HStack } from "@astryxdesign/core/HStack";
import { Selector } from "@astryxdesign/core/Selector";
import { Spinner } from "@astryxdesign/core/Spinner";
import { Table, type TableColumn } from "@astryxdesign/core/Table";
import { Text } from "@astryxdesign/core/Text";
import { Token } from "@astryxdesign/core/Token";
import { VStack } from "@astryxdesign/core/VStack";
import type { Role, WarBatRuns, WarEvalRow, WarFile, WarGroup, WarMetric, WarRow } from "@/src/data/types";
import type { JsonState } from "@/src/data/useJson";
import { DASH, fmtCount, fmtRuns, fmtWar, fmtWar2, fmtWarMae } from "@/src/format";
import {
  BAT_RUNS_PARTS,
  BAT_WAR_SUMMARY,
  FULL_TIME_NAME,
  FULL_TIME_SHORT,
  PIT_WAR_NOTE,
  PIT_WAR_STABILITY,
  PIT_WAR_TITLE,
  RUNS_NOT_EXACT,
  WAR_FACTS,
  WAR_TEST_INTRO,
  WAR_WHAT,
  runsPerWinNote,
  WAR_STAR_LOSS_REASON,
} from "@/src/warText";

// ---------------------------------------------------------------------------
// 이름표 (warText에 없는 시험 표 문구 — 시안 SA-8)
// ---------------------------------------------------------------------------

const ROLE_LABEL: Record<Role, string> = { bat: "타자", pit: "투수" };

/** 시험 표 여섯 칸 — 이 순서로 보여 준다 (SA-8) */
const CELLS: ReadonlyArray<{ role: Role; metric: WarMetric; group: WarGroup; who: string }> = [
  { role: "bat", metric: "war", group: "all", who: "전원" },
  { role: "pit", metric: "war", group: "all", who: "전원" },
  { role: "bat", metric: "war_full", group: "regulars", who: "주전급" },
  { role: "pit", metric: "war_full", group: "regulars", who: "주전급" },
  { role: "bat", metric: "war", group: "war2plus", who: "실제 WAR 2 이상" },
  { role: "pit", metric: "war", group: "war2plus", who: "실제 WAR 2 이상" },
];

const METRIC_WHAT: Record<WarMetric, string> = { war: "예측 WAR", war_full: FULL_TIME_SHORT };

const BASE_PREV = "작년 그대로";
const BASE_LEAGUE = "리그 평균 비율";

const RESULT_TEXT = {
  win: "두 기준선보다 낮음",
  losePrev: `${BASE_PREV}에 짐`,
  loseLeague: `${BASE_LEAGUE}에 짐`,
  loseBoth: "두 기준선에 짐",
} as const;

const NOT_TESTED = "아직 시험하지 않았습니다";
const FULL_TIME_REASON = `그래서 순위표에 ${FULL_TIME_NAME}을 같이 둡니다.`;

const COUNT_WORD: Record<number, string> = { 1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯" };
const countWord = (n: number) => COUNT_WORD[n] ?? fmtCount(n);

// ---------------------------------------------------------------------------
// 순수 계산
// ---------------------------------------------------------------------------

type CellRow = {
  id: string;
  what: string;
  who: string;
  row: WarEvalRow;
  /** 진 기준선 (이 모델 오차가 기준선보다 작지 않으면 진 것 — 같으면 진 것으로 센다) */
  lost: Array<{ name: string; value: number }>;
  /** 이 모델 오차와 두 기준선 오차가 모두 있어 이기고 진 것을 가릴 수 있는가 */
  isScored: boolean;
};

function cellRows(evaluations: readonly WarEvalRow[], season: number): CellRow[] {
  const out: CellRow[] = [];
  for (const cell of CELLS) {
    const row = evaluations.find(
      (r) => r.target_season === season && r.role === cell.role && r.metric === cell.metric && r.group === cell.group,
    );
    if (row === undefined) continue;
    const lost: CellRow["lost"] = [];
    if (row.mae != null) {
      const bases: Array<[string, number | null]> = [
        [BASE_PREV, row.baseline_mae_prev],
        [BASE_LEAGUE, row.baseline_mae_league],
      ];
      for (const [name, value] of bases) if (value != null && !(row.mae < value)) lost.push({ name, value });
    }
    out.push({
      id: `${cell.role}-${cell.metric}-${cell.group}`,
      isScored: row.mae != null && row.baseline_mae_prev != null && row.baseline_mae_league != null,
      what: `${ROLE_LABEL[cell.role]} ${METRIC_WHAT[cell.metric]}`,
      who: `${cell.who} ${fmtCount(row.n)}명`,
      row,
      lost,
    });
  }
  return out;
}

function resultText(c: CellRow): string {
  // 값이 빠진 칸은 이겼다고 적지 않는다 (진 기준선이 있으면 그것만은 적는다)
  if (c.lost.length === 0) return c.isScored ? RESULT_TEXT.win : DASH;
  if (c.lost.length === 2) return RESULT_TEXT.loseBoth;
  return c.lost[0]!.name === BASE_PREV ? RESULT_TEXT.losePrev : RESULT_TEXT.loseLeague;
}

/** 결과 문장 — 이긴 칸 수, 진 칸은 숫자와 함께 그대로 (D36 — 숨기지 않는다) */
function summaryText(cells: CellRow[], isFinal: boolean): string {
  // 진 칸은 값이 일부 빠져도 적고, 이긴 칸은 두 기준선이 다 있을 때만 센다
  const scored = cells.filter((c) => c.isScored || c.lost.length > 0);
  if (scored.length === 0) return "";
  const lost = scored.filter((c) => c.lost.length > 0);
  const total = countWord(scored.length);
  if (lost.length === 0) return `${total} 칸 모두 두 기준선보다 오차가 작았습니다.`;
  const sentences: string[] = [];
  const won = scored.length - lost.length;
  sentences.push(
    won > 0
      ? `${total} 칸 중 ${countWord(won)} 칸에서 두 기준선보다 오차가 작았습니다.`
      : `${total} 칸 중 두 기준선보다 오차가 작은 칸은 없었습니다.`,
  );
  const list = lost
    .map((c) => {
      const vs = c.lost.map((b) => `${b.name} ${fmtWarMae(b.value)}`).join(" · ");
      return `${c.what}(${c.who}) ${fmtWarMae(c.row.mae)} 대 ${vs}`;
    })
    .join(", ");
  sentences.push(`기준선보다 오차가 작지 않은 칸 — ${list}.`);
  if (lost.some((c) => c.row.group === "war2plus")) sentences.push(WAR_STAR_LOSS_REASON);
  if (isFinal) sentences.push(FULL_TIME_REASON);
  return sentences.join(" ");
}

const runsSum = (r: WarBatRuns) => BAT_RUNS_PARTS.reduce((sum, p) => sum + r[p.key], 0);

/** 「WAR은 어떻게 계산되나」의 예 — 투타 겸업 + 구성 항목이 있는 선수 중 예측 WAR이 가장 큰 선수 (없으면 null) */
function exampleText(war: WarFile, prevSeason: number): string | null {
  let best: WarRow | null = null;
  for (const row of war.rows) {
    if (row.roles !== "both" || row.prev_bat_runs === null) continue;
    if (best === null || row.war > best.war) best = row;
  }
  if (best === null || best.prev_bat_runs === null) return null;
  const batWar = best.history.find((h) => h.season === prevSeason)?.war_bat;
  if (batWar == null) return null;
  const runs = best.prev_bat_runs;
  const parts = BAT_RUNS_PARTS.map((p) => `${p.name} ${fmtRuns(runs[p.key])}`).join(", ");
  // 선수별 점 → 승 비율은 적지 않는다 (E29)
  return `예: ${best.player.name} ${prevSeason} 타자 WAR ${fmtWar2(batWar)} = ${fmtWar(runsSum(runs))}점(${parts})`;
}

// ---------------------------------------------------------------------------
// 화면 조각
// ---------------------------------------------------------------------------

const TEST_COLUMNS: TableColumn<CellRow>[] = [
  { key: "what", header: "무엇을", renderCell: (c) => <Text weight="bold">{c.what}</Text> },
  {
    key: "who",
    header: "누구에게",
    renderCell: (c) => (
      <Text color="secondary" hasTabularNumbers>
        {c.who}
      </Text>
    ),
  },
  {
    key: "mae",
    header: "이 모델",
    align: "end",
    renderCell: (c) => (
      <Text weight="bold" hasTabularNumbers>
        {fmtWarMae(c.row.mae)}
      </Text>
    ),
  },
  {
    key: "prev",
    header: BASE_PREV,
    align: "end",
    renderCell: (c) => <Text hasTabularNumbers>{fmtWarMae(c.row.baseline_mae_prev)}</Text>,
  },
  {
    key: "league",
    header: BASE_LEAGUE,
    align: "end",
    renderCell: (c) => <Text hasTabularNumbers>{fmtWarMae(c.row.baseline_mae_league)}</Text>,
  },
  {
    key: "result",
    header: "결과",
    // 이긴 칸·진 칸을 색만이 아니라 글로도 (N10)
    renderCell: (c) =>
      c.lost.length === 0 && c.isScored ? (
        <Text weight="bold" color="accent">
          {resultText(c)}
        </Text>
      ) : (
        <Text>{resultText(c)}</Text>
      ),
  },
];

function TestSection({ war }: { war: WarFile }) {
  const ft = war.final_test;
  const finalSeason = ft?.target_season ?? Math.max(...war.meta.input_seasons);
  // 목표 시즌: 최종 시험 시즌(2026, 시험 전이어도 넣는다) + 과거 검증 시즌, 최신 먼저
  const seasons = useMemo(
    () => [...new Set([finalSeason, ...war.evaluations.map((r) => r.target_season)])].sort((a, b) => b - a),
    [finalSeason, war.evaluations],
  );
  const [picked, setPicked] = useState<string>(String(finalSeason));
  const season = Number(picked);
  const isFinal = season === finalSeason;
  const notTested = isFinal && ft === null;
  const cells = useMemo(() => (notTested ? [] : cellRows(war.evaluations, season)), [notTested, war.evaluations, season]);
  const summary = summaryText(cells, isFinal);

  return (
    <VStack as="section" gap={4} aria-labelledby="war-test">
      <HStack gap={3} hAlign="between" vAlign="center" wrap="wrap">
        <Heading id="war-test" level={2}>
          {`WAR 예측 — ${finalSeason} 시험`}
        </Heading>
        {ft !== null && <Token label={`고정한 설정으로 한 번 · 다시 돌림 ${fmtCount(ft.reruns.length)}번`} />}
      </HStack>
      <Text as="p" color="secondary">
        {WAR_TEST_INTRO}
      </Text>
      <HStack gap={4} wrap="wrap" vAlign="end">
        <Selector
          label="WAR 목표 시즌"
          options={seasons.map((s) => String(s))}
          value={picked}
          onChange={(v) => setPicked(v)}
        />
      </HStack>
      {/* 다시 돌렸으면 기록 파일의 사유를 그대로 둔다 — 숨기지 않는다 (D36, 기존 최종 검증과 같은 장치) */}
      {ft !== null &&
        ft.reruns.map((r) => (
          <Text key={r.ran_at} as="p" type="supporting" color="secondary">
            {`기록된 사유: ${r.reason}`}
          </Text>
        ))}
      {notTested ? (
        <EmptyState title={`${finalSeason} 시즌은 ${NOT_TESTED}`} isCompact />
      ) : (
        <>
          <Table<CellRow>
            data={cells}
            columns={TEST_COLUMNS}
            idKey="id"
            aria-labelledby="war-test"
            emptyState={<EmptyState title="이 시즌의 WAR 검증 결과가 없습니다" isCompact />}
          />
          {summary !== "" && (
            <Text as="p" hasTabularNumbers>
              {summary}
            </Text>
          )}
        </>
      )}
    </VStack>
  );
}

type StabilityRow = { id: string; name: string; corr: readonly number[]; isUsed: boolean };

const STABILITY_COLUMNS: TableColumn<StabilityRow>[] = [
  {
    key: "name",
    header: "투수 WAR 정의",
    renderCell: (r) =>
      r.isUsed ? (
        <Text weight="bold" color="accent">
          {r.name}
        </Text>
      ) : (
        <Text>{r.name}</Text>
      ),
  },
  {
    key: "y2024",
    header: "2024",
    align: "end",
    renderCell: (r) => (
      <Text weight={r.isUsed ? "bold" : "normal"} hasTabularNumbers>
        {fmtWar2(r.corr[0])}
      </Text>
    ),
  },
  {
    key: "y2025",
    header: "2025",
    align: "end",
    renderCell: (r) => (
      <Text weight={r.isUsed ? "bold" : "normal"} hasTabularNumbers>
        {fmtWar2(r.corr[1])}
      </Text>
    ),
  },
];

const STABILITY_ROWS: StabilityRow[] = PIT_WAR_STABILITY.rows.map((r) => ({
  id: r.name,
  name: r.name,
  corr: r.corr,
  isUsed: r.isUsed,
}));

function HowSection({ war }: { war: WarFile }) {
  const prevSeason = Math.max(...war.meta.input_seasons);
  const example = exampleText(war, prevSeason);

  return (
    <VStack as="section" gap={4} aria-labelledby="war-how">
      <Heading id="war-how" level={2}>
        WAR은 어떻게 계산되나
      </Heading>
      <Text as="p" color="secondary">
        {WAR_WHAT}
      </Text>
      <Grid columns={{ minWidth: 300, max: 2 }} gap={4}>
        <Card>
          <VStack gap={3}>
            <Heading level={3}>타자 WAR</Heading>
            {/* 구성 항목(점)의 합 ÷ 1승에 필요한 점수 — 이름만, 순서는 BAT_RUNS_PARTS */}
            <HStack gap={2} wrap="wrap" vAlign="center">
              {BAT_RUNS_PARTS.map((p, i) => (
                <HStack key={p.key} gap={2} vAlign="center">
                  {i > 0 && <Text color="secondary">+</Text>}
                  <Token label={p.name} />
                </HStack>
              ))}
            </HStack>
            <Text as="p" type="supporting" color="secondary" hasTabularNumbers>
              {`점수 합 ÷ 1승에 필요한 점수 (${runsPerWinNote(war.runs_per_win)})`}
            </Text>
            <Text as="p">{BAT_WAR_SUMMARY}</Text>
            {example !== null && (
              <Text as="p" hasTabularNumbers>
                {example}
              </Text>
            )}
            {example !== null && (
              <Text as="p" type="supporting" color="secondary">
                {RUNS_NOT_EXACT}
              </Text>
            )}
          </VStack>
        </Card>
        <Card>
          <VStack gap={3}>
            <Heading level={3}>{PIT_WAR_TITLE}</Heading>
            <Text as="p">{PIT_WAR_NOTE}</Text>
            <Text id="war-pit-stability" type="label" color="secondary">
              {PIT_WAR_STABILITY.caption}
            </Text>
            <Table<StabilityRow>
              data={STABILITY_ROWS}
              columns={STABILITY_COLUMNS}
              idKey="id"
              aria-labelledby="war-pit-stability"
            />
            <Text as="p" type="supporting" color="secondary">
              {PIT_WAR_STABILITY.note}
            </Text>
          </VStack>
        </Card>
      </Grid>
      <Grid columns={{ minWidth: 260, max: 3 }} gap={4}>
        {WAR_FACTS.map((f) => (
          <Card key={f.title} variant="muted">
            <VStack gap={2}>
              <Heading level={3}>{f.title}</Heading>
              <Text as="p" color="secondary">
                {f.body}
              </Text>
            </VStack>
          </Card>
        ))}
      </Grid>
    </VStack>
  );
}

// ---------------------------------------------------------------------------
// 구역
// ---------------------------------------------------------------------------

export function WarValidation({ state }: { state: JsonState<WarFile> }) {
  if (state.status === "loading") {
    return (
      <HStack hAlign="center" paddingBlock={8}>
        <Spinner size="lg" label="WAR 검증 결과를 불러오는 중…" aria-label="WAR 검증 결과를 불러오는 중" />
      </HStack>
    );
  }
  if (state.status === "error") {
    return (
      <Banner
        status="error"
        title="WAR 검증 결과를 불러오지 못했습니다"
        description={
          state.notFound
            ? "WAR 자료 파일이 아직 없습니다. 위의 여섯 지표 검증은 그대로 볼 수 있습니다."
            : "잠시 뒤 다시 열어 주세요. 위의 여섯 지표 검증은 그대로 볼 수 있습니다."
        }
      />
    );
  }
  return (
    <>
      <TestSection war={state.data} />
      <HowSection war={state.data} />
    </>
  );
}
