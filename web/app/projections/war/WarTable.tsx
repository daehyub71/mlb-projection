"use client";

// WAR 순위 데스크톱 표 (SA-6) — # · 선수 · 역할 · 2027 예측 WAR* · 풀타임 환산 · 2026 WAR · 예측 출전*
// 정렬은 표 머리가 아니라 위의 세그먼트(예측 WAR / 풀타임 환산 WAR)가 맡는다. 숫자는 모두 src/format.ts로 쓴다.
import { Badge } from "@astryxdesign/core/Badge";
import { HStack } from "@astryxdesign/core/HStack";
import { Link } from "@astryxdesign/core/Link";
import { List, ListItem } from "@astryxdesign/core/List";
import { pixel, proportional, Table, type TableColumn } from "@astryxdesign/core/Table";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import type { WarRoles } from "@/src/data/types";
import { DASH, fmtWar, fmtWarSplit, warRoleLabel } from "@/src/format";
import { playerHref } from "@/src/site";
import { FULL_TIME_ADJUSTED, FULL_TIME_NAME, FULL_TIME_SHORT, fullTimeUnits } from "@/src/warText";
import { Figure, PlayerAvatar } from "../cells";
import { playingTime, warPlayerMeta, type WarView } from "./warRows";

/** 역할 배지 색 — DESIGN §4-1(타 짙은 청록 · 투 보라 · 투타 자홍)에 가장 가까운 Badge 변형 */
const ROLE_VARIANT: Record<WarRoles, "teal" | "purple" | "pink"> = { bat: "teal", pit: "purple", both: "pink" };

export function WarRoleBadge({ roles }: { roles: WarRoles }) {
  return <Badge label={warRoleLabel(roles)} variant={ROLE_VARIANT[roles]} />;
}

/** "현역 아님" (D23) — active=false일 때만 */
export function WarInactiveBadge({ view }: { view: WarView }) {
  return view.active ? null : <Badge label="현역 아님" variant="neutral" />;
}

/** 선수 칸 — 사진 원 · 이름(상세 링크) · 포지션 · 팀 (예측표 PlayerCell과 같은 모양) */
function WarPlayerCell({ view }: { view: WarView }) {
  const meta = warPlayerMeta(view);
  return (
    <HStack gap={3} vAlign="center">
      <PlayerAvatar playerId={view.id} name={view.name} />
      <VStack gap={0.5}>
        <HStack gap={2} vAlign="center" wrap="wrap">
          <Link href={playerHref(view.id)} color="primary" weight="bold">
            {view.name}
          </Link>
          <WarInactiveBadge view={view} />
        </HStack>
        <Text type="supporting" color="secondary">
          {meta === "" ? DASH : meta}
        </Text>
      </VStack>
    </HStack>
  );
}

/** 2027 예측 WAR — 가장 큰 숫자, 투타 겸업은 아래에 "타 3.9 + 투 2.3" (D34) */
export function WarFigure({ view, size = "lg" }: { view: WarView; size?: "lg" | "xl" }) {
  const split = fmtWarSplit(view.warBat, view.warPit);
  return (
    <VStack gap={0.5}>
      <Text size={size} weight="bold" hasTabularNumbers>
        {fmtWar(view.war)}
      </Text>
      {split !== "" && (
        <Text type="supporting" color="secondary" hasTabularNumbers>
          {split}
        </Text>
      )}
    </VStack>
  );
}

function warColumns(seasons: { target: number; prev: number }): TableColumn<WarView>[] {
  return [
    {
      key: "rank",
      header: "#",
      width: pixel(56),
      renderCell: (v) => <Figure isMuted>{String(v.rank)}</Figure>,
    },
    {
      key: "name",
      header: "선수",
      width: proportional(3, { minWidth: 178 }),
      renderCell: (v) => <WarPlayerCell view={v} />,
    },
    {
      key: "roles",
      header: "역할",
      width: pixel(72),
      renderCell: (v) => <WarRoleBadge roles={v.roles} />,
    },
    {
      key: "war",
      header: `${seasons.target} 예측 WAR*`,
      width: pixel(128),
      renderCell: (v) => <WarFigure view={v} />,
    },
    {
      key: "warFull",
      header: FULL_TIME_SHORT,
      width: pixel(96),
      align: "end",
      renderCell: (v) => <Figure>{fmtWar(v.warFull)}</Figure>,
    },
    {
      key: "prevWar",
      header: `${seasons.prev} WAR`,
      width: pixel(100),
      align: "end",
      renderCell: (v) => <Figure isMuted>{fmtWar(v.prevWar)}</Figure>,
    },
    {
      key: "playingTime",
      header: "예측 출전*",
      width: pixel(136),
      align: "end",
      // 예측 출전이 둘 다 없는 행(이론상)은 빈 칸 대신 —
      renderCell: (v) => <Figure isMuted>{playingTime(v) || DASH}</Figure>,
    },
  ];
}

export function WarTable({
  rows,
  seasons,
  label,
}: {
  rows: WarView[];
  seasons: { target: number; prev: number };
  label: string;
}) {
  return (
    <Table<WarView>
      aria-label={label}
      data={rows}
      columns={warColumns(seasons)}
      idKey="id"
      density="balanced"
      hasHover
    />
  );
}

/** 옆 상자 제목 (SA-6) */
export const FULL_TIME_BOX_TITLE = `${FULL_TIME_NAME}로 정렬하면`;

/**
 * 옆 상자 「풀타임 환산 WAR로 정렬하면」 (SA-6) — 전원 기준 war_full 상위 5명, 각 "예측 X.X · 출전".
 * 지금 정렬과 상관없이 같은 다섯 명이다 — 짧게 뛴 선수가 풀타임으로 얼마나 올라오는지 보여 준다.
 */
export function FullTimeBox({
  top,
  fullTime,
  prevSeason,
}: {
  top: WarView[];
  fullTime: { pa: number; sp_ip: number; rp_ip: number };
  prevSeason: number;
}) {
  return (
    <VStack as="aside" aria-label={FULL_TIME_BOX_TITLE} gap={3} width={288} className="shrink-0">
      <Text weight="bold">{FULL_TIME_BOX_TITLE}</Text>
      <List hasDividers edge="flush" aria-label={`${FULL_TIME_NAME} 상위 ${top.length}명`}>
        {top.map((v) => (
          <ListItem
            key={v.id}
            itemKey={v.id}
            href={playerHref(v.id)}
            label={
              <HStack gap={2} hAlign="between" vAlign="center">
                <Text weight="bold">{v.name}</Text>
                <Text weight="bold" hasTabularNumbers>
                  {fmtWar(v.warFull)}
                </Text>
              </HStack>
            }
            description={
              <Text type="supporting" color="secondary" hasTabularNumbers>
                {[`예측 ${fmtWar(v.war)}`, playingTime(v)].filter((p) => p !== "").join(" · ")}
              </Text>
            }
          />
        ))}
      </List>
      <Text as="p" type="supporting" color="secondary">
        {`풀타임 환산 = 같은 실력으로 ${fullTimeUnits(fullTime)}을 뛴다면. ${FULL_TIME_ADJUSTED}입니다. ${prevSeason}에 짧게 뛴 선수는 예측 WAR이 낮게 나오므로 둘을 같이 봅니다.`}
      </Text>
    </VStack>
  );
}
