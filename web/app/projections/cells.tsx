"use client";

// 예측표의 칸 부품 — 선수(사진·이름·포지션·팀), 비율 지표 ± 범위, 리그 평균 대비 막대 (DESIGN §4 v0.2 표시 규칙)
import { Avatar } from "@astryxdesign/core/Avatar";
import { Badge } from "@astryxdesign/core/Badge";
import { HStack } from "@astryxdesign/core/HStack";
import { Link } from "@astryxdesign/core/Link";
import { ProgressBar } from "@astryxdesign/core/ProgressBar";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import { DASH, fmtCount, fmtDiff, fmtRate } from "@/src/format";
import { initials, photosEnabled, playerPhotoUrl } from "@/src/photo";
import { playerHref } from "@/src/site";
import type { BatView, PitView } from "./rows";

type AnyView = BatView | PitView;

/** 리그 평균 대비 막대의 구간 — .600~1.000 (DESIGN §4) */
const BAR_MIN = 0.6;
const BAR_MAX = 1.0;

/** 사진은 40px 원, 고해상도 화면을 위해 두 배 폭으로 받는다 */
const PHOTO_WIDTH = 80;

/**
 * 선수 사진 원 (D27). MLB 서버 사진을 그대로 불러오고, 못 불러오면(또는 사진을 끄면) 이니셜 배지.
 * 이름 옆에 붙는 장식이라 보조 기술에서는 숨긴다 — 이름은 바로 옆 링크가 읽어 준다.
 */
export function PlayerAvatar({ playerId, name }: { playerId: number; name: string }) {
  // Avatar는 첫 단어와 끝 단어로 이니셜을 만든다("Bobby Witt Jr." → "BJ").
  // 계약 initials()("BW")와 같게 하려고 그 글자를 띄어 써서 넘긴다
  const letters = initials(name).split("").join(" ");
  return (
    <Avatar
      src={photosEnabled() ? playerPhotoUrl(playerId, PHOTO_WIDTH) : undefined}
      name={letters}
      size={40}
      tooltip={false}
      aria-hidden
    />
  );
}

/**
 * "DH · Houston Astros" — 빈 값은 빼고, 다 비면 —.
 * 마지막 MLB 시즌(D23)은 "현역 아님"이거나 직전 시즌에 뛰지 않은 선수에게 붙인다 — 팀 이름이 옛 팀이라는 걸 알 수 있게.
 * 직전 시즌에 뛴 현역 선수는 모두 같은 해라서 적지 않는다(SA-1).
 */
export function playerMeta(view: AnyView, withAge: boolean): string {
  const parts = [view.position, view.team, withAge && view.age !== null ? `${fmtCount(view.age)}세` : null].filter(
    (p): p is string => p !== null && p !== "",
  );
  // 연도는 숫자 표기(천 단위 쉼표)를 쓰지 않는다
  if (!view.active || !view.playedPrev) parts.push(`마지막 MLB 시즌 ${view.lastSeason}`);
  return parts.length === 0 ? DASH : parts.join(" · ");
}

/** "현역 아님" (D23) — active=false일 때만. true에는 아무것도 붙이지 않는다 */
export function InactiveBadge({ view }: { view: AnyView }) {
  return view.active ? null : <Badge label="현역 아님" variant="neutral" />;
}

/** 표의 선수 칸 — 사진 · 이름(상세 링크) · 포지션·팀 */
export function PlayerCell({ view }: { view: AnyView }) {
  return (
    <HStack gap={3} vAlign="center">
      <PlayerAvatar playerId={view.id} name={view.name} />
      <VStack gap={0.5}>
        <HStack gap={2} vAlign="center" wrap="wrap">
          <Link href={playerHref(view.id)} color="primary" weight="bold">
            {view.name}
          </Link>
          <InactiveBadge view={view} />
        </HStack>
        <Text type="supporting" color="secondary">
          {playerMeta(view, false)}
        </Text>
      </VStack>
    </HStack>
  );
}

/** 2027 예측 비율 지표 — 큰 숫자 + 아래에 작게 ± 과거 오차 80% 범위 (D29) */
export function ProjectionFigure({ value, band }: { value: string; band: string | null }) {
  return (
    <VStack gap={0.5}>
      <Text size="lg" weight="bold" hasTabularNumbers>
        {value}
      </Text>
      {band !== null && (
        <Text type="supporting" color="secondary" hasTabularNumbers>
          {band}
        </Text>
      )}
    </VStack>
  );
}

/** 표 숫자 칸 — 자릿수를 맞춘다 */
export function Figure({ children, isMuted = false }: { children: string; isMuted?: boolean }) {
  return (
    <Text hasTabularNumbers color={isMuted ? "secondary" : undefined}>
      {children}
    </Text>
  );
}

function barPercent(ops: number): number {
  return Math.min(100, Math.max(0, ((ops - BAR_MIN) / (BAR_MAX - BAR_MIN)) * 100));
}

/**
 * 리그 평균 대비 — .600~1.000 막대에 리그 평균 눈금, 옆에 차이(+.220).
 * 막대는 보이는 모양일 뿐이라 inert로 감싸 탭 정지·보조 기술에서 뺀다. 뜻은 옆의 차이 글자가 전한다.
 */
export function LeagueDiff({ ops, diff, league }: { ops: number; diff: number | null; league: number | null }) {
  if (diff === null || league === null) return <Figure isMuted>{DASH}</Figure>;
  return (
    <HStack gap={2} vAlign="center">
      <VStack inert width={112}>
        <ProgressBar
          label="리그 평균 대비 위치"
          isLabelHidden
          value={barPercent(ops)}
          marks={[{ value: barPercent(league), label: `리그 평균 ${fmtRate(league)}` }]}
        />
      </VStack>
      <Figure>{fmtDiff("ops", diff)}</Figure>
    </HStack>
  );
}
