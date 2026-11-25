"use client";

// 예측표 열 정의 (SA-1) — 타자: 선수 · 나이 · 사용 기록 · 2027 OPS · 리그 평균 대비 · PA* · HR* · 2026 OPS
//                        투수: 선수 · 나이 · 사용 기록 · 2027 ERA · 2027 K−BB% · IP* · 2026 ERA
// 열 key가 곧 정렬 key다 (rows.ts의 BatSortKey·PitSortKey). 숫자는 모두 src/format.ts로 쓴다.
import { pixel, proportional, type TableColumn } from "@astryxdesign/core/Table";
import { Text } from "@astryxdesign/core/Text";
import { fmtBand, fmtCount, fmtEra, fmtIp, fmtPct, fmtRate } from "@/src/format";
import { Figure, LeagueDiff, PlayerCell, ProjectionFigure } from "./cells";
import type { BatView, PitView } from "./rows";

/** 머리 글에 들어가는 시즌 — 목표 시즌(2027)과 직전 입력 시즌(2026)은 meta에서 읽는다 */
export interface SeasonLabels {
  target: number;
  prev: number;
}

const band = (metric: string, v: number | null): string | null => (v === null ? null : fmtBand(metric, v));

function sampleText(nSeasons: number, amount: string): string {
  return `${fmtCount(nSeasons)}시즌 · ${amount}`;
}

export function batColumns(seasons: SeasonLabels, league: number | null): TableColumn<BatView>[] {
  return [
    {
      key: "name",
      header: "선수",
      width: proportional(3, { minWidth: 240 }),
      sortable: true,
      renderCell: (v) => <PlayerCell view={v} />,
    },
    {
      key: "age",
      header: "나이",
      width: pixel(72),
      sortable: true,
      renderCell: (v) => <Figure>{fmtCount(v.age)}</Figure>,
    },
    {
      key: "samplePa",
      header: "사용 기록",
      width: pixel(156),
      sortable: true,
      renderCell: (v) => (
        <Text color="secondary" hasTabularNumbers>
          {sampleText(v.nSeasons, `${fmtCount(v.samplePa)}타석`)}
        </Text>
      ),
    },
    {
      key: "ops",
      header: `${seasons.target} OPS`,
      width: pixel(112),
      sortable: true,
      renderCell: (v) => <ProjectionFigure value={fmtRate(v.ops)} band={band("ops", v.opsBand)} />,
    },
    {
      key: "opsDiff",
      header: league === null ? "리그 평균 대비" : `리그 평균(${fmtRate(league)}) 대비`,
      width: pixel(200),
      renderCell: (v) => <LeagueDiff ops={v.ops} diff={v.opsDiff} league={league} />,
    },
    {
      key: "pa",
      header: "PA*",
      width: pixel(80),
      align: "end",
      sortable: true,
      renderCell: (v) => <Figure>{fmtCount(v.pa)}</Figure>,
    },
    {
      key: "hr",
      header: "HR*",
      width: pixel(80),
      align: "end",
      sortable: true,
      renderCell: (v) => <Figure>{fmtCount(v.hr)}</Figure>,
    },
    {
      key: "prevOps",
      header: `${seasons.prev} OPS`,
      width: pixel(104),
      align: "end",
      sortable: true,
      renderCell: (v) => <Figure isMuted>{fmtRate(v.prevOps)}</Figure>,
    },
  ];
}

export function pitColumns(seasons: SeasonLabels): TableColumn<PitView>[] {
  return [
    {
      key: "name",
      header: "선수",
      width: proportional(3, { minWidth: 240 }),
      sortable: true,
      renderCell: (v) => <PlayerCell view={v} />,
    },
    {
      key: "age",
      header: "나이",
      width: pixel(72),
      sortable: true,
      renderCell: (v) => <Figure>{fmtCount(v.age)}</Figure>,
    },
    {
      key: "sampleIp",
      header: "사용 기록",
      width: pixel(156),
      sortable: true,
      renderCell: (v) => (
        <Text color="secondary" hasTabularNumbers>
          {sampleText(v.nSeasons, `${fmtIp(v.sampleIp)}이닝`)}
        </Text>
      ),
    },
    {
      key: "era",
      header: `${seasons.target} ERA`,
      width: pixel(112),
      sortable: true,
      renderCell: (v) => <ProjectionFigure value={fmtEra(v.era)} band={band("era", v.eraBand)} />,
    },
    {
      key: "kbb",
      header: `${seasons.target} K−BB%`,
      width: pixel(128),
      sortable: true,
      renderCell: (v) => <ProjectionFigure value={fmtPct(v.kbb)} band={band("k_bb_pct", v.kbbBand)} />,
    },
    {
      key: "ip",
      header: "IP*",
      width: pixel(88),
      align: "end",
      sortable: true,
      renderCell: (v) => <Figure>{fmtIp(v.ip)}</Figure>,
    },
    {
      key: "prevEra",
      header: `${seasons.prev} ERA`,
      width: pixel(104),
      align: "end",
      sortable: true,
      renderCell: (v) => <Figure isMuted>{fmtEra(v.prevEra)}</Figure>,
    },
  ];
}
