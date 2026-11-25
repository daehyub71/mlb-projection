"use client";

// 수비 과거 기록 표 (SA-2, SPEC D21) — 예측은 하지 않는다. OAA와 FRV는 단위가 달라 더하지 않는다
import { Table } from "@astryxdesign/core/Table";
import { pixel, proportional, type TableColumn } from "@astryxdesign/core/Table/utils";
import type { FieldingSeason } from "@/src/data/types";
import { DASH, fmtCount } from "@/src/format";
import { fmtFrv, fmtOaa } from "./text";

type Row = Record<string, string>;
type FrvKey = "frv_range" | "frv_arm" | "frv_dp" | "frv_framing" | "frv_blocking" | "frv_throwing";

// FRV 세부 — 값이 하나라도 있는 열만 보인다 (포수는 프레이밍·블로킹·도루 저지)
const FRV_PARTS: ReadonlyArray<{ key: FrvKey; header: string }> = [
  { key: "frv_range", header: "FRV 범위" },
  { key: "frv_arm", header: "FRV 어깨" },
  { key: "frv_dp", header: "FRV 병살" },
  { key: "frv_framing", header: "FRV 프레이밍" },
  { key: "frv_blocking", header: "FRV 블로킹" },
  { key: "frv_throwing", header: "FRV 도루 저지" },
];

/** "SS 3,450 · 2B 120" — 아웃 많은 포지션부터 */
function positionsCell(outs: FieldingSeason["outs_by_pos"]): string {
  const entries = Object.entries(outs)
    .filter((e): e is [string, number] => typeof e[1] === "number")
    .sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return DASH;
  return entries.map(([pos, n]) => `${pos} ${fmtCount(n)}`).join(" · ");
}

/** labelledBy: 구역 제목의 id (표 이름, N10) */
export function FieldingTable({ seasons, labelledBy }: { seasons: FieldingSeason[]; labelledBy: string }) {
  const sorted = [...seasons].sort((a, b) => a.season - b.season);
  const parts = FRV_PARTS.filter((p) => sorted.some((s) => s[p.key] != null));
  const columns: TableColumn<Row>[] = [
    { key: "season", header: "시즌", width: pixel(72) },
    { key: "positions", header: "포지션 (아웃 수)", width: proportional(3, { minWidth: 180 }) },
    { key: "oaa", header: "OAA", align: "end" },
    { key: "frv_total", header: "FRV 합계", align: "end" },
    ...parts.map((p) => ({ key: p.key, header: p.header, align: "end" as const })),
  ];
  const rows: Row[] = sorted.map((s) => ({
    id: String(s.season),
    season: String(s.season),
    positions: positionsCell(s.outs_by_pos),
    oaa: fmtOaa(s.oaa),
    frv_total: fmtFrv(s.frv_total),
    ...Object.fromEntries(parts.map((p) => [p.key, fmtFrv(s[p.key])])),
  }));
  return <Table data={rows} columns={columns} idKey="id" density="compact" aria-labelledby={labelledBy} />;
}
