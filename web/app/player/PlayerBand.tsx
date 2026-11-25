"use client";

// 선수 머리 띠 (SA-2) — 히어로 이미지를 깐 띠 위에 사진·이름·한 줄 정보·칩·역할 전환·2027 예측 카드 3개
import Image from "next/image";
import { Avatar } from "@astryxdesign/core/Avatar";
import { Card } from "@astryxdesign/core/Card";
import { Grid } from "@astryxdesign/core/Grid";
import { Heading } from "@astryxdesign/core/Heading";
import { HStack } from "@astryxdesign/core/HStack";
import { Link } from "@astryxdesign/core/Link";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import { StackItem } from "@astryxdesign/core/Layout";
import { Text } from "@astryxdesign/core/Text";
import { Token } from "@astryxdesign/core/Token";
import { VStack } from "@astryxdesign/core/VStack";
import type { PlayerInfo, Role } from "@/src/data/types";
import { initials, PHOTO_CREDIT, photosEnabled, playerPhotoUrl } from "@/src/photo";
import { seasonAge } from "@/src/age";
import { fmtCount } from "@/src/format";
import { handednessText, positionText, ROLE_NOUN, sampleText, windowText, type ProjectionCard, type RoleDetail } from "./text";

interface PlayerBandProps {
  player: PlayerInfo;
  targetSeason: number;
  current: RoleDetail;
  /** 투타 겸업이면 두 역할 (F12) */
  roles: Role[];
  onRoleChange: (role: Role) => void;
  /** 최근 팀 — 마지막 기록 행의 팀 */
  team: string | null;
  cards: ProjectionCard[];
}

/** 사진 원 (D27). 설정을 끄거나 불러오지 못하면 Avatar가 이니셜로 바꾼다 */
function PlayerPhoto({ player }: { player: PlayerInfo }) {
  const showPhoto = photosEnabled(); // 렌더 때 읽는다 — 설정 하나로 바로 끈다
  // Avatar는 name의 첫·끝 낱말로 이니셜을 만든다. 계약의 initials()("Bobby Witt Jr." → "BW")를 쓰려고
  // 글자를 띄어 넘기고, 화면 낭독기 이름은 alt(선수 이름)로 준다.
  const badge = initials(player.name).split("").join(" ");
  return (
    <VStack gap={1} hAlign="center">
      <Avatar
        src={showPhoto ? playerPhotoUrl(player.player_id) : undefined}
        name={badge}
        alt={player.name}
        size={96}
        tooltip={false}
        className="ring-2 ring-accent-bg"
      />
      {showPhoto && <Text type="supporting" color="secondary">{PHOTO_CREDIT}</Text>}
    </VStack>
  );
}

function ProjectionCardView({ card, targetSeason }: { card: ProjectionCard; targetSeason: number }) {
  return (
    <Card padding={5}>
      <VStack gap={2}>
        <Text type="label" weight="bold" color="accent">
          {`${targetSeason} ${card.label}`}
          {card.isStarred ? "*" : ""}
        </Text>
        <HStack gap={2} vAlign="end" wrap="wrap">
          {/* SA-2: 카드 숫자와 선수 이름이 같은 크기(display-3) — 첫 화면 히어로 제목(display-2)보다 한 단계 작다 */}
          <Text type="display-3" weight="bold" hasTabularNumbers>
            {card.value}
          </Text>
          {card.band !== null && (
            <Text type="large" hasTabularNumbers>
              {card.band}
            </Text>
          )}
        </HStack>
        <Text type="supporting" color="secondary" hasTabularNumbers>
          {card.sub}
        </Text>
      </VStack>
    </Card>
  );
}

export function PlayerBand({ player, targetSeason, current, roles, onRoleChange, team, cards }: PlayerBandProps) {
  const age = seasonAge(player.birth_date, targetSeason);
  const info = [
    positionText(player.primary_position),
    handednessText(player.bats, player.throws),
    age === null ? null : `${targetSeason}년 ${fmtCount(age)}세`,
    team,
  ].filter((part): part is string => part !== null && part !== "");

  return (
    <VStack as="section" aria-label="선수 정보" className="relative isolate overflow-hidden">
      {/* 히어로 C 이미지(장식) — 왼쪽은 바탕색으로 덮어 글자 대비를 지킨다 (N10) */}
      <Image src="/images/hero.jpg" alt="" fill preload sizes="100vw" className="-z-10 object-cover object-right opacity-40" />
      <VStack hAlign="center" className="bg-linear-to-r from-body via-body/90 to-body/60">
        <VStack gap={5} paddingBlock={6} paddingInline={6} width="100%" maxWidth={1200}>
          <HStack as="nav" aria-label="경로" gap={1} vAlign="center">
            <Link href="/" type="supporting">
              예측표
            </Link>
            <Text type="supporting" color="secondary" aria-hidden="true">
              ›
            </Text>
            <Text type="supporting" color="secondary">
              {ROLE_NOUN[current.role]}
            </Text>
          </HStack>

          <HStack gap={5} vAlign="center" wrap="wrap">
            <PlayerPhoto player={player} />
            <StackItem size="fill">
              <VStack gap={2}>
                <Heading level={1} type="display-3" textWrap="balance">
                  {player.name}
                </Heading>
                {info.length > 0 && <Text hasTabularNumbers>{info.join(" · ")}</Text>}
                <HStack gap={2} wrap="wrap">
                  {!player.active && <Token label="현역 아님" color="orange" />}
                  <Token label={`마지막 MLB 시즌 ${player.last_mlb_season}`} />
                  <Token label={`사용한 기록 ${windowText(current.detail.window)} · ${sampleText(current)}`} />
                </HStack>
              </VStack>
            </StackItem>
            {roles.length > 1 && (
              <SegmentedControl label="역할" value={current.role} onChange={(value) => onRoleChange(value as Role)} size="lg">
                {roles.map((role) => (
                  <SegmentedControlItem key={role} value={role} label={ROLE_NOUN[role]} />
                ))}
              </SegmentedControl>
            )}
          </HStack>

          <Grid columns={{ minWidth: 220, max: 3 }} gap={3}>
            {cards.map((card) => (
              <ProjectionCardView key={card.metric} card={card} targetSeason={targetSeason} />
            ))}
          </Grid>
        </VStack>
      </VStack>
    </VStack>
  );
}
