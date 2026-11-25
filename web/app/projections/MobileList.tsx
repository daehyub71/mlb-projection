"use client";

// 좁은 화면(SA-3, 390px)의 예측표 — 표와 같은 순서의 선수 목록. 한 줄이 상세로 가는 링크다.
// Astryx 규칙(촘촘한 자료는 행으로)에 따라 카드 대신 구분선 있는 List를 쓴다.
import { Grid } from "@astryxdesign/core/Grid";
import { HStack } from "@astryxdesign/core/HStack";
import { List, ListItem } from "@astryxdesign/core/List";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import { fmtBand, fmtCount, fmtEra, fmtIp, fmtPct, fmtRate } from "@/src/format";
import { playerHref } from "@/src/site";
import { InactiveBadge, PlayerAvatar, playerMeta } from "./cells";
import type { SeasonLabels } from "./columns";
import type { BatView, PitView } from "./rows";

interface Stat {
  label: string;
  value: string;
  band?: string | null;
  isLead?: boolean;
}

/** 지표 한 칸 — 이름 위, 숫자 아래 (2027 비율 지표가 가장 크다) */
function StatBlock({ stat }: { stat: Stat }) {
  return (
    <VStack gap={0.5}>
      <Text type="supporting" weight="bold" color={stat.isLead ? "accent" : "secondary"}>
        {stat.label}
      </Text>
      <Text size={stat.isLead ? "xl" : "lg"} weight="bold" hasTabularNumbers>
        {stat.value}
      </Text>
      {stat.band != null && (
        <Text type="supporting" color="secondary" hasTabularNumbers>
          {stat.band}
        </Text>
      )}
    </VStack>
  );
}

function PlayerRow({
  view,
  stats,
  footer,
}: {
  view: BatView | PitView;
  stats: Stat[];
  footer: string;
}) {
  return (
    <ListItem
      itemKey={view.id}
      href={playerHref(view.id)}
      align="start"
      startContent={<PlayerAvatar playerId={view.id} name={view.name} />}
      label={
        <HStack gap={2} vAlign="center" wrap="wrap">
          <Text weight="bold">{view.name}</Text>
          <InactiveBadge view={view} />
        </HStack>
      }
      description={
        <VStack gap={2} paddingBlockStart={0.5}>
          <Text type="supporting" color="secondary" hasTabularNumbers>
            {playerMeta(view, true)}
          </Text>
          <Grid columns={3} gap={2}>
            {stats.map((s) => (
              <StatBlock key={s.label} stat={s} />
            ))}
          </Grid>
          <Text type="supporting" color="secondary" hasTabularNumbers>
            {footer}
          </Text>
        </VStack>
      }
    />
  );
}

const band = (metric: string, v: number | null): string | null => (v === null ? null : fmtBand(metric, v));

export function BatList({ rows, seasons, label }: { rows: BatView[]; seasons: SeasonLabels; label: string }) {
  return (
    <List hasDividers edge="flush" aria-label={label}>
      {rows.map((v) => (
        <PlayerRow
          key={v.id}
          view={v}
          stats={[
            { label: `${seasons.target} OPS`, value: fmtRate(v.ops), band: band("ops", v.opsBand), isLead: true },
            { label: "PA*", value: fmtCount(v.pa) },
            { label: "HR*", value: fmtCount(v.hr) },
          ]}
          footer={`${fmtCount(v.nSeasons)}시즌 기록 · ${seasons.prev} OPS ${fmtRate(v.prevOps)}`}
        />
      ))}
    </List>
  );
}

export function PitList({ rows, seasons, label }: { rows: PitView[]; seasons: SeasonLabels; label: string }) {
  return (
    <List hasDividers edge="flush" aria-label={label}>
      {rows.map((v) => (
        <PlayerRow
          key={v.id}
          view={v}
          stats={[
            { label: `${seasons.target} ERA`, value: fmtEra(v.era), band: band("era", v.eraBand), isLead: true },
            { label: "K−BB%", value: fmtPct(v.kbb), band: band("k_bb_pct", v.kbbBand) },
            { label: "IP*", value: fmtIp(v.ip) },
          ]}
          footer={`${fmtCount(v.nSeasons)}시즌 기록 · ${seasons.prev} ERA ${fmtEra(v.prevEra)}`}
        />
      ))}
    </List>
  );
}
