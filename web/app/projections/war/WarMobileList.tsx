"use client";

// 좁은 화면(SA-9, 390px)의 WAR 순위 — 선수 카드: 순위 · 사진 원 · 이름 · 포지션 · 팀 · 역할 배지,
// 2027 WAR*(투타는 분해 한 줄) · 풀타임 환산 둘만. 2026 WAR · 예측 출전은 선수 상세에서 본다(사용자 "카드를 더 간단히").
// 예측표 모바일 목록(MobileList.tsx)처럼 구분선 있는 List로 그린다. 한 줄이 상세로 가는 링크다.
import { Grid } from "@astryxdesign/core/Grid";
import { HStack } from "@astryxdesign/core/HStack";
import { List, ListItem } from "@astryxdesign/core/List";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import { DASH, fmtWar } from "@/src/format";
import { playerHref } from "@/src/site";
import { FULL_TIME_SHORT } from "@/src/warText";
import { PlayerAvatar } from "../cells";
import { WarFigure, WarInactiveBadge, WarRoleBadge } from "./WarTable";
import { warPlayerMeta, type WarView } from "./warRows";

export function WarMobileList({ rows, target, label }: { rows: WarView[]; target: number; label: string }) {
  return (
    <List hasDividers edge="flush" aria-label={label}>
      {rows.map((v) => {
        const meta = warPlayerMeta(v);
        return (
          <ListItem
            key={v.id}
            itemKey={v.id}
            href={playerHref(v.id)}
            align="start"
            startContent={
              <HStack gap={2} vAlign="center">
                <Text weight="bold" color="secondary" hasTabularNumbers>
                  {String(v.rank)}
                </Text>
                <PlayerAvatar playerId={v.id} name={v.name} />
              </HStack>
            }
            label={
              <HStack gap={2} vAlign="center" wrap="wrap">
                <Text weight="bold">{v.name}</Text>
                <WarRoleBadge roles={v.roles} />
                <WarInactiveBadge view={v} />
              </HStack>
            }
            description={
              <VStack gap={2} paddingBlockStart={0.5}>
                <Text type="supporting" color="secondary">
                  {meta === "" ? DASH : meta}
                </Text>
                <Grid columns={2} gap={2}>
                  <VStack gap={0.5}>
                    <Text type="supporting" weight="bold" color="accent">
                      {`${target} WAR*`}
                    </Text>
                    <WarFigure view={v} size="xl" />
                  </VStack>
                  <VStack gap={0.5}>
                    <Text type="supporting" weight="bold" color="secondary">
                      {FULL_TIME_SHORT}
                    </Text>
                    <Text size="lg" weight="bold" hasTabularNumbers>
                      {fmtWar(v.warFull)}
                    </Text>
                  </VStack>
                </Grid>
              </VStack>
            }
          />
        );
      })}
    </List>
  );
}
