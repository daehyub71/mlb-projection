"use client";

// 선수 상세 WAR 구역 (F22, 시안 SA-7 — DESIGN §4-1 WAR 표시 규칙)
// 선수 상세를 받은 뒤에만 그려진다 — 그때 war.json을 한 번 받는다(N6). 2026 미출전 선수는 안내 한 줄(D42).
import { useMemo } from "react";
import { Chart, ChartAxis, ChartGrid, bar, referenceLine, type TooltipSeriesValue } from "@astryxdesign/charts";
import { Card } from "@astryxdesign/core/Card";
import { Grid } from "@astryxdesign/core/Grid";
import { Heading } from "@astryxdesign/core/Heading";
import { Spinner } from "@astryxdesign/core/Spinner";
import { Table } from "@astryxdesign/core/Table";
import { pixel, type TableColumn } from "@astryxdesign/core/Table/utils";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import { loadWar } from "@/src/data/load";
import type { PlayerDetail, WarFile, WarRow, WarSeason } from "@/src/data/types";
import { useJson } from "@/src/data/useJson";
import { fmtCount, fmtRuns, fmtWar, fmtWar2, fmtWarSplit } from "@/src/format";
import { findWarRow } from "@/src/war";
import {
  BAT_RUNS_PARTS,
  FULL_TIME_ADJUSTED,
  FULL_TIME_NAME,
  PIT_WAR_FIP_COL,
  PIT_WAR_NOTE,
  PIT_WAR_RA9_COL,
  PIT_WAR_TITLE,
  RUNS_NOT_EXACT,
  WAR_ABOUT,
  WAR_LABEL,
  WAR_NOT_IN_FILE,
  runsPerWinNote,
} from "@/src/warText";

// 색은 테마 토큰 — 합의 시안 SA-7 그대로 타자는 자홍(강조색), 투수는 보라
const BAT_COLOR = "var(--color-accent)";
const PIT_COLOR = "var(--color-data-purple-3)";
const FULL_COLOR = "var(--color-text-secondary)";

// 차트 데이터 키는 숨겨 둔 자료 표의 열 이름이 된다 — 한국어로 둔다
const X_KEY = "시즌";
const BAT_KEY = "타자 WAR";
const PIT_KEY = "투수 WAR(FIP 기반)";
const BAT_PROJ_KEY = "예측 타자 WAR";
const PIT_PROJ_KEY = "예측 투수 WAR";

function finite(v: number | null | undefined): v is number {
  return v != null && Number.isFinite(v);
}

/** 숫자 뒤 조사 은/는 — 끝자리 읽기(영·일·삼·육·칠·팔)에 받침이 있으면 「은」 ("3.69는", "4.18은") */
function topicParticle(text: string): string {
  return /[013678]$/.test(text) ? "은" : "는";
}

/** 오래된 순 시즌 */
function sortedHistory(history: WarSeason[]): WarSeason[] {
  return [...history].sort((a, b) => a.season - b.season);
}

interface RoleFlags {
  hasBat: boolean;
  hasPit: boolean;
}

function roleFlags(row: WarRow): RoleFlags {
  return { hasBat: row.roles !== "pit", hasPit: row.roles !== "bat" };
}

// ---------------------------------------------------------------------------
// 카드 셋 — ① 예측 WAR* ② 풀타임 환산 WAR ③ 예측 출전*
// ---------------------------------------------------------------------------

function WarCard({ label, value, lines }: { label: string; value: string; lines: string[] }) {
  return (
    <Card padding={5}>
      <VStack gap={2}>
        <Text type="label" weight="bold" color="accent">
          {label}
        </Text>
        <Text type="display-3" weight="bold" hasTabularNumbers>
          {value}
        </Text>
        {lines.map((line) => (
          <Text key={line} type="supporting" color="secondary" hasTabularNumbers>
            {line}
          </Text>
        ))}
      </VStack>
    </Card>
  );
}

/** ① 아래 줄: "2026 실제 6.6 (3.7 + 2.9) · 보수적 예상" — 투타가 아니면 괄호 없이 */
function prevLine(row: WarRow, prevSeason: number, flags: RoleFlags): string {
  if (!finite(row.prev_war)) return "보수적 예상";
  let text = `${prevSeason} 실제 ${fmtWar(row.prev_war)}`;
  const prev = row.history.find((s) => s.season === prevSeason);
  if (flags.hasBat && flags.hasPit && prev !== undefined && finite(prev.war_bat) && finite(prev.war_pit)) {
    text += ` (${fmtWar(prev.war_bat)} + ${fmtWar(prev.war_pit)})`;
  }
  return `${text} · 보수적 예상`;
}

/** ② 기준 줄: 그 선수 역할의 기준만 (숫자는 war.json full_time) */
function fullTimeLines(row: WarRow, full: WarFile["full_time"], flags: RoleFlags): string[] {
  const batBasis = `타자 ${full.pa}타석`;
  const pitBasis = row.slot === "rp" ? `불펜 ${full.rp_ip}이닝` : `선발 ${full.sp_ip}이닝`;
  const lines: string[] = [];
  if (flags.hasBat && flags.hasPit) {
    lines.push(`${batBasis} ${fmtWar(row.war_full_bat)} + ${pitBasis} ${fmtWar(row.war_full_pit)}`);
    lines.push(`${FULL_TIME_ADJUSTED} — 두 역할 모두 풀타임인 경우는 드물다`);
  } else {
    lines.push(`${flags.hasBat ? batBasis : pitBasis} 기준`);
    lines.push(FULL_TIME_ADJUSTED);
  }
  return lines;
}

/**
 * ③ 예측 출전: 큰 숫자는 한 역할만("500타석" — 투타 겸업이면 타석), 투수 몫은 아래 줄
 * "투수 77이닝 · 선발로 분류"(SA-7). 이닝도 fmtCount 반올림.
 */
function playingTime(row: WarRow, flags: RoleFlags): { value: string; lines: string[] } {
  const slot = row.slot === "rp" ? "불펜으로 분류" : "선발로 분류";
  const pit = flags.hasPit && finite(row.ip) ? `${fmtCount(row.ip)}이닝` : null;
  if (flags.hasBat && finite(row.pa)) {
    return { value: `${fmtCount(row.pa)}타석`, lines: pit === null ? [] : [`투수 ${pit} · ${slot}`] };
  }
  return pit === null ? { value: fmtCount(null), lines: [] } : { value: pit, lines: [slot] };
}

/** ③ 아래 줄: 선수 상세 기록의 2026 타석 · 이닝 */
function prevPlayingTime(detail: PlayerDetail, prevSeason: number, flags: RoleFlags): string | null {
  const parts: string[] = [];
  const bat = flags.hasBat ? detail.bat?.history.find((s) => s.season === prevSeason) : undefined;
  const pit = flags.hasPit ? detail.pit?.history.find((s) => s.season === prevSeason) : undefined;
  if (bat !== undefined && finite(bat.pa)) parts.push(`${fmtCount(bat.pa)}타석`);
  if (pit !== undefined && finite(pit.ip)) parts.push(`${fmtCount(pit.ip)}이닝`);
  return parts.length > 0 ? `${prevSeason} 실제 ${parts.join(" · ")}` : null;
}

function WarCards({ detail, row, war, prevSeason }: { detail: PlayerDetail; row: WarRow; war: WarFile; prevSeason: number }) {
  const flags = roleFlags(row);
  const target = war.meta.target_season;
  const split = fmtWarSplit(row.war_bat, row.war_pit);
  const predLines = [...(split !== "" ? [split] : []), prevLine(row, prevSeason, flags)];
  const prevTime = prevPlayingTime(detail, prevSeason, flags);
  const time = playingTime(row, flags);
  return (
    <Grid columns={{ minWidth: 260, max: 3 }} gap={3}>
      <WarCard label={`${target} 예측 WAR*`} value={fmtWar(row.war)} lines={predLines} />
      <WarCard label={FULL_TIME_NAME} value={fmtWar(row.war_full)} lines={fullTimeLines(row, war.full_time, flags)} />
      <WarCard label="예측 출전*" value={time.value} lines={[...time.lines, ...(prevTime === null ? [] : [prevTime])]} />
    </Grid>
  );
}

// ---------------------------------------------------------------------------
// 시즌별 WAR — 타 · 투 쌓은 막대 + 2027 예측 막대 + 풀타임 환산 가로선
// ---------------------------------------------------------------------------

type ChartRow = Record<string, string | number | null>;

/** 화면 낭독기와 화면 모두에 읽어 주는 값 설명 (Chart subtitle → svg <desc>) */
function describeSeasons(row: WarRow, seasons: WarSeason[], flags: RoleFlags, target: number): string {
  const pieces = seasons.map((s) => {
    const parts: string[] = [];
    if (flags.hasBat && finite(s.war_bat)) parts.push(`타자 ${fmtWar(s.war_bat)}`);
    if (flags.hasPit && finite(s.war_pit)) parts.push(`투수 ${fmtWar(s.war_pit)}`);
    return `${s.season} ${parts.length > 0 ? parts.join(" + ") : "기록 없음"}`;
  });
  const split = fmtWarSplit(row.war_bat, row.war_pit);
  let text = `${target} 예측 ${fmtWar(row.war)}${split !== "" ? ` (${split})` : ""} · ${FULL_TIME_NAME} ${fmtWar(row.war_full)}`;
  if (pieces.length > 0) text = `${pieces.join(" · ")} → ${text}`;
  return text;
}

/**
 * 세로축 범위 — 쌓은 막대의 위·아래 합과 풀타임 환산 가로선을 모두 담는다.
 * Astryx Chart는 referenceLine을 범위 계산에 넣지 않아, 풀타임 환산이 막대보다 크면(실데이터 635명) 선이 잘린다.
 */
function warDomain(rows: ChartRow[], keys: string[], full: number | null): [number, number] {
  let min = 0;
  let max = 0;
  for (const r of rows) {
    let up = 0;
    let down = 0;
    for (const key of keys) {
      const v = r[key];
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      if (v >= 0) up += v;
      else down += v;
    }
    max = Math.max(max, up);
    min = Math.min(min, down);
  }
  if (finite(full)) {
    max = Math.max(max, full);
    min = Math.min(min, full);
  }
  const pad = max > min ? (max - min) * 0.12 : 1;
  return [min < 0 ? min - pad : 0, max + pad];
}

function SeasonWarChart({ row, target }: { row: WarRow; target: number }) {
  const flags = roleFlags(row);
  const seasons = useMemo(() => sortedHistory(row.history), [row.history]);
  const targetLabel = `${target} 예측`;
  const rows = useMemo<ChartRow[]>(
    () => [
      ...seasons.map((s) => ({
        [X_KEY]: String(s.season),
        [BAT_KEY]: finite(s.war_bat) ? s.war_bat : null,
        [PIT_KEY]: finite(s.war_pit) ? s.war_pit : null,
        [BAT_PROJ_KEY]: null,
        [PIT_PROJ_KEY]: null,
      })),
      {
        [X_KEY]: targetLabel,
        [BAT_KEY]: null,
        [PIT_KEY]: null,
        [BAT_PROJ_KEY]: finite(row.war_bat) ? row.war_bat : null,
        [PIT_PROJ_KEY]: finite(row.war_pit) ? row.war_pit : null,
      },
    ],
    [seasons, row.war_bat, row.war_pit, targetLabel],
  );
  // 마크 정의는 차트마다 새로 만든다 (Astryx: 한 정의를 여러 Chart에 다시 쓰지 않는다)
  const series = useMemo(
    () => [
      ...(flags.hasBat ? [bar(BAT_KEY, { label: BAT_KEY, stack: "war", color: BAT_COLOR })] : []),
      ...(flags.hasPit ? [bar(PIT_KEY, { label: PIT_KEY, stack: "war", color: PIT_COLOR })] : []),
      ...(flags.hasBat ? [bar(BAT_PROJ_KEY, { label: `${target} ${BAT_PROJ_KEY}`, stack: "war", color: BAT_COLOR, opacity: 0.5 })] : []),
      ...(flags.hasPit ? [bar(PIT_PROJ_KEY, { label: `${target} ${PIT_PROJ_KEY}`, stack: "war", color: PIT_COLOR, opacity: 0.5 })] : []),
      ...(finite(row.war_full)
        ? [referenceLine({ y: row.war_full, label: `풀타임 환산 ${fmtWar(row.war_full)}`, color: FULL_COLOR, labelPosition: "start" as const })]
        : []),
    ],
    [flags.hasBat, flags.hasPit, target, row.war_full],
  );
  const domain = useMemo(
    () => warDomain(rows, [BAT_KEY, PIT_KEY, BAT_PROJ_KEY, PIT_PROJ_KEY], row.war_full),
    [rows, row.war_full],
  );

  const renderTooltip = (xValue: unknown, values: TooltipSeriesValue[]) => {
    const known = values.filter((v) => typeof v.value === "number");
    return (
      <VStack gap={1}>
        <Text weight="semibold">{String(xValue)}</Text>
        {known.length === 0 ? (
          <Text type="supporting" color="secondary">
            기록 없음
          </Text>
        ) : (
          known.map((v) => (
            <Text key={v.key} type="supporting" hasTabularNumbers>
              {v.label} {fmtWar(v.value as number)}
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
      margin={{ left: 48 }}
      yDomain={domain}
      title={`시즌별 ${flags.hasBat && flags.hasPit ? "타자 · 투수" : flags.hasBat ? "타자" : "투수"} WAR과 ${targetLabel}`}
      subtitle={describeSeasons(row, seasons, flags, target)}
      grid={<ChartGrid horizontal />}
      axes={
        <>
          <ChartAxis position="bottom" />
          <ChartAxis position="left" tickFormat={(v) => fmtWar(Number(v))} />
        </>
      }
      legend
      tooltip={{ render: renderTooltip }}
    />
  );
}

// ---------------------------------------------------------------------------
// 2026 타자 WAR 구성 항목 (E29)
// ---------------------------------------------------------------------------

type TextRow = Record<string, string>;

/** 구성 합: 부호 없는 1자리 + "점" (음수만 − U+2212) — fmtRuns에서 + 만 뗀다 */
function runsTotal(v: number): string {
  return `${fmtRuns(v).replace(/^\+/, "")}점`;
}

const TOTAL_ID = "total";

/** 합계 줄은 표 마지막 줄로 둔다 (시안 SA-7) — 글자를 굵게 */
function renderRunsCell(r: TextRow, key: "name" | "note" | "runs") {
  if (r.id !== TOTAL_ID) return r[key];
  return (
    <Text weight="bold" color={key === "note" ? "secondary" : undefined} hasTabularNumbers>
      {r[key]}
    </Text>
  );
}

const RUNS_COLUMNS: TableColumn<TextRow>[] = [
  { key: "name", header: "항목", width: pixel(120), renderCell: (r) => renderRunsCell(r, "name") },
  { key: "note", header: "설명", renderCell: (r) => renderRunsCell(r, "note") },
  { key: "runs", header: "점", align: "end", width: pixel(140), renderCell: (r) => renderRunsCell(r, "runs") },
];

function BatRunsSection({ row, war, prevSeason }: { row: WarRow; war: WarFile; prevSeason: number }) {
  const runs = row.prev_bat_runs;
  if (runs === null) return null;
  const prevBat = row.history.find((s) => s.season === prevSeason)?.war_bat ?? null;
  const total = BAT_RUNS_PARTS.reduce((sum, part) => sum + runs[part.key], 0);
  const hasPrevBat = finite(prevBat);
  const data: TextRow[] = [
    ...BAT_RUNS_PARTS.map((part) => ({
      id: part.key,
      name: part.name,
      note: part.note,
      runs: fmtRuns(runs[part.key]),
    })),
    {
      id: TOTAL_ID,
      name: "합계",
      note: runsPerWinNote(war.runs_per_win),
      // 2026 타자 WAR이 없으면(계약상 없어야 하지만) "—승" 대신 점만
      runs: hasPrevBat ? `${runsTotal(total)} → ${fmtWar2(prevBat)}승` : runsTotal(total),
    },
  ];
  const id = "player-war-runs";
  const warText = fmtWar2(prevBat);
  const title = hasPrevBat
    ? `${prevSeason} 타자 WAR ${warText}${topicParticle(warText)} 이렇게 나왔다`
    : `${prevSeason} 타자 WAR은 이렇게 나왔다`;
  return (
    <Card padding={5}>
      <VStack as="section" gap={3} aria-labelledby={id}>
        <Heading level={3} id={id}>
          {title}
        </Heading>
        <Table data={data} columns={RUNS_COLUMNS} idKey="id" density="compact" aria-labelledby={id} />
        <Text type="supporting" color="secondary">
          {RUNS_NOT_EXACT}
        </Text>
      </VStack>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 투수 WAR — FIP 기반 (D40)
// ---------------------------------------------------------------------------

const PIT_COLUMNS: TableColumn<TextRow>[] = [
  { key: "season", header: "시즌", width: pixel(96) },
  { key: "fip", header: PIT_WAR_FIP_COL, align: "end" },
  { key: "ra9", header: PIT_WAR_RA9_COL, align: "end" },
];

function PitWarSection({ row }: { row: WarRow }) {
  if (!row.history.some((s) => finite(s.war_pit))) return null;
  const data: TextRow[] = sortedHistory(row.history).map((s) => ({
    id: String(s.season),
    season: String(s.season),
    fip: fmtWar2(s.war_pit),
    ra9: fmtWar2(s.war_pit_ra9),
  }));
  const id = "player-war-pit";
  return (
    <Card padding={5}>
      <VStack as="section" gap={3} aria-labelledby={id}>
        <Heading level={3} id={id}>
          {PIT_WAR_TITLE}
        </Heading>
        <Table data={data} columns={PIT_COLUMNS} idKey="id" density="compact" aria-labelledby={id} />
        <Text type="supporting" color="secondary">
          {PIT_WAR_NOTE}
        </Text>
      </VStack>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 구역 전체
// ---------------------------------------------------------------------------

function WarBody({ detail, row, war }: { detail: PlayerDetail; row: WarRow; war: WarFile }) {
  const target = war.meta.target_season;
  const prevSeason = target - 1;
  // WAR 구역 하나(h2) 아래에 시즌별 차트 · 구성 항목 · 투수 WAR(h3)을 둔다
  return (
    <VStack as="section" gap={6} aria-labelledby="player-war">
      <VStack gap={3}>
        <Heading level={2} id="player-war">
          {WAR_LABEL}
        </Heading>
        <Text type="supporting" color="secondary">
          {WAR_ABOUT}
        </Text>
        <WarCards detail={detail} row={row} war={war} prevSeason={prevSeason} />
      </VStack>
      <VStack as="section" gap={3} aria-labelledby="player-war-seasons">
        <Heading level={3} id="player-war-seasons">
          시즌별 WAR
        </Heading>
        <Card padding={5}>
          <SeasonWarChart row={row} target={target} />
        </Card>
      </VStack>
      <Grid columns={{ minWidth: 360, max: 2 }} gap={4}>
        <BatRunsSection row={row} war={war} prevSeason={prevSeason} />
        <PitWarSection row={row} />
      </Grid>
    </VStack>
  );
}

/** 선수 상세 아래 WAR 구역 — PlayerDetail이 선수 상세를 받은 뒤에만 붙인다 */
export function WarSection({ detail }: { detail: PlayerDetail }) {
  const state = useJson(() => loadWar(), "war");

  if (state.status === "loading") {
    return (
      <VStack paddingBlock={6} hAlign="center">
        <Spinner size="md" label="WAR 기록을 불러오는 중…" aria-label="WAR 기록을 불러오는 중" />
      </VStack>
    );
  }
  if (state.status === "error") {
    return (
      <Card padding={5}>
        <Text color="secondary">WAR 기록을 불러오지 못했습니다. 잠시 뒤 다시 시도해 주세요.</Text>
      </Card>
    );
  }
  const row = findWarRow(state.data, detail.player.player_id);
  if (row === null) {
    return <Text color="secondary">{WAR_NOT_IN_FILE}</Text>;
  }
  return <WarBody detail={detail} row={row} war={state.data} />;
}
