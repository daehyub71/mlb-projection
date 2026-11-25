"use client";

// 첫 화면 히어로 (SA-1·SA-3, DS5) — 히어로 C 사진 위에 머리말·큰 제목·한 줄 소개·대상 수 칩·단추
// 사진은 장식(alt="")이고, 글자는 바탕색 그라데이션 위에 놓여 대비를 지킨다(N10).
import Image from "next/image";
import { Button } from "@astryxdesign/core/Button";
import { Heading } from "@astryxdesign/core/Heading";
import { HStack } from "@astryxdesign/core/HStack";
import { Text } from "@astryxdesign/core/Text";
import { Token } from "@astryxdesign/core/Token";
import { VStack } from "@astryxdesign/core/VStack";
import { fmtCount } from "@/src/format";

export interface HeroCounts {
  /** 예측 대상 타자 수 — 받은 타자 표의 행 수 */
  batters: number | null;
  /** 투수 수 — 투수 표는 처음 투수 탭을 누를 때 받으므로 그 전에는 null (칩을 숨긴다) */
  pitchers: number | null;
  /** 입력 시즌 (meta.input_seasons) */
  inputSeasons: readonly number[] | null;
}

const HERO_TITLE_ID = "hero-title";

function seasonRange(seasons: readonly number[]): string {
  return `${Math.min(...seasons)}~${Math.max(...seasons)}`;
}

export function Hero({ counts, tableAnchor }: { counts: HeroCounts; tableAnchor: string }) {
  const chips = [
    counts.batters === null ? null : `예측 대상 타자 ${fmtCount(counts.batters)}명`,
    counts.pitchers === null ? null : `투수 ${fmtCount(counts.pitchers)}명`,
    counts.inputSeasons === null || counts.inputSeasons.length === 0
      ? null
      : `입력 ${seasonRange(counts.inputSeasons)}`,
  ].filter((c): c is string => c !== null);

  return (
    // 이름은 제목(h1)에서 — 헤더 메뉴의 "소개" 페이지와 헷갈리지 않게
    <VStack as="section" aria-labelledby={HERO_TITLE_ID} className="relative isolate min-h-90 md:min-h-128">
      <Image
        src="/images/hero.jpg"
        alt=""
        fill
        preload
        sizes="100vw"
        className="-z-10 object-cover object-center md:object-right"
      />
      {/* 글자 뒤 바탕색 그라데이션 — 좁은 화면은 아래로, 넓은 화면은 왼쪽으로 짙어진다 */}
      <VStack
        className="grow justify-end bg-linear-to-b from-transparent via-body/70 to-body md:justify-center md:bg-linear-to-r md:from-body/95 md:via-body/75 md:to-transparent"
        paddingBlock={10}
      >
        {/* 표 영역과 같은 1200 열에 글자를 맞춘다 */}
        <VStack width="100%" maxWidth={1200} paddingInline={6} className="mx-auto">
          <VStack gap={4} maxWidth={640}>
            <Text type="label" color="accent" weight="bold">
              2027 PROJECTIONS · JUST FOR FUN
            </Text>
            {/* 한글은 낱말 단위로 줄을 바꾼다 (break-keep) — "선 / 수"처럼 쪼개지 않게 */}
            <Heading level={1} id={HERO_TITLE_ID} type="display-2" textWrap="balance" className="break-keep">
              2027 MLB 선수 성적 예측
            </Heading>
            <Text as="p" type="large" color="secondary">
              최근 5시즌 기록으로 계산한, 재미로 보는 예측입니다.
            </Text>
            {chips.length > 0 && (
              <HStack gap={2} wrap="wrap" className="hidden md:flex">
                {chips.map((chip) => (
                  <Token key={chip} label={chip} />
                ))}
              </HStack>
            )}
            <HStack gap={3} wrap="wrap" paddingBlockStart={2} className="hidden md:flex">
              <Button label="예측표 보기" variant="primary" size="lg" href={`#${tableAnchor}`} />
              <Button label="얼마나 맞았나" variant="secondary" size="lg" href="/validation/" />
            </HStack>
          </VStack>
        </VStack>
      </VStack>
    </VStack>
  );
}
