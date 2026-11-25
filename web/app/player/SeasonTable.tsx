"use client";

// 최근 5시즌 기록 표 (SA-2, SPEC D14) — 숫자는 format.ts로 미리 글자로 바꿔 넣는다
import { Table } from "@astryxdesign/core/Table";
import { pixel, type TableColumn } from "@astryxdesign/core/Table/utils";
import type { BatSeason, PitSeason } from "@/src/data/types";
import { DASH, fmtCount, fmtEra, fmtIp, fmtMetric, fmtPct, fmtRate } from "@/src/format";

type Row = Record<string, string>;

/** 표에 보이는 시즌: 기록의 마지막 5시즌, 오래된 순 (F14 "최대 5시즌") */
export function recentSeasons<T extends { season: number }>(history: T[]): T[] {
  return [...history].sort((a, b) => a.season - b.season).slice(-5);
}

function seasonCell(s: { season: number; is_short_season: boolean }): string {
  return s.is_short_season ? `${s.season} · 60경기` : String(s.season);
}

const num = (key: string, header: string): TableColumn<Row> => ({ key, header, align: "end" });

// 글자가 긴 열만 고정 폭 — 비례 열에 큰 최소 폭을 주면 표 전체의 최소 폭이 그만큼 불어난다
const LEAD_COLUMNS: TableColumn<Row>[] = [
  { key: "season", header: "시즌", width: pixel(96) },
  { key: "team", header: "팀", width: pixel(160) },
  num("age", "나이"),
];

const BAT_COLUMNS: TableColumn<Row>[] = [
  ...LEAD_COLUMNS,
  num("pa", "PA"),
  num("hr", "HR"),
  num("obp", "OBP"),
  num("slg", "SLG"),
  num("ops", "OPS"),
  num("woba", "wOBA"),
  num("xwoba", "xwOBA"),
  // 긴 머리글 두 개는 고정 폭 — 비례 열로 두면 "Barrel / %"·"Hard / Hit%"처럼 두 줄로 꺾인다
  { key: "barrel", header: "Barrel%", align: "end", width: pixel(88) },
  { key: "hardHit", header: "Hard Hit%", align: "end", width: pixel(96) },
  num("ev", "EV"), // 평균 타구 속도(mph, D14)
  { key: "kbb", header: "K% / BB%", align: "end", width: pixel(104) },
];

const PIT_COLUMNS: TableColumn<Row>[] = [
  ...LEAD_COLUMNS,
  num("g", "G"),
  num("gs", "GS"),
  num("ip", "IP"),
  num("era", "ERA"),
  num("fip", "FIP"),
  num("k", "K%"),
  num("bb", "BB%"),
  num("kbb", "K−BB%"),
  num("xera", "xERA"),
  { key: "xwoba", header: "xwOBA 허용", align: "end", width: pixel(96) },
  num("whiff", "Whiff%"),
];

function batRow(s: BatSeason): Row {
  return {
    id: String(s.season),
    season: seasonCell(s),
    team: s.team_name ?? DASH,
    age: fmtCount(s.age),
    pa: fmtCount(s.pa),
    hr: fmtCount(s.hr),
    obp: fmtRate(s.obp),
    slg: fmtRate(s.slg),
    ops: fmtRate(s.ops),
    woba: fmtRate(s.woba),
    xwoba: fmtRate(s.xwoba),
    barrel: fmtPct(s.barrel_pct),
    hardHit: fmtPct(s.hard_hit_pct),
    ev: fmtMetric("ev_avg", s.ev_avg),
    kbb: `${fmtPct(s.k_pct)} / ${fmtPct(s.bb_pct)}`,
  };
}

function pitRow(s: PitSeason): Row {
  return {
    id: String(s.season),
    season: seasonCell(s),
    team: s.team_name ?? DASH,
    age: fmtCount(s.age),
    g: fmtCount(s.g),
    gs: fmtCount(s.gs),
    ip: fmtIp(s.ip),
    era: fmtEra(s.era),
    fip: fmtEra(s.fip),
    k: fmtPct(s.k_pct),
    bb: fmtPct(s.bb_pct),
    kbb: fmtPct(s.k_bb_pct),
    xera: fmtEra(s.xera),
    xwoba: fmtRate(s.xwoba_against),
    whiff: fmtPct(s.whiff_pct),
  };
}

// labelledBy: 구역 제목의 id — 표와 옆으로 미는 영역이 기본 이름("Table") 대신 그 제목으로 읽힌다 (N10)
export function BatSeasonTable({ history, labelledBy }: { history: BatSeason[]; labelledBy: string }) {
  return (
    <Table
      data={recentSeasons(history).map(batRow)}
      columns={BAT_COLUMNS}
      idKey="id"
      density="compact"
      aria-labelledby={labelledBy}
    />
  );
}

export function PitSeasonTable({ history, labelledBy }: { history: PitSeason[]; labelledBy: string }) {
  return (
    <Table
      data={recentSeasons(history).map(pitRow)}
      columns={PIT_COLUMNS}
      idKey="id"
      density="compact"
      aria-labelledby={labelledBy}
    />
  );
}
