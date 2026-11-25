"use client";

// 선수 상세 본문 (F14, SA-2) — 주소 ?id=를 읽어 players/{id}.json 하나만 받는다 (DS1, 정적 export)
import { useState } from "react";
import { useSearchParams } from "next/navigation";
import { ArrowRightIcon, ExclamationTriangleIcon, InformationCircleIcon, MagnifyingGlassIcon } from "@heroicons/react/24/outline";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { Grid } from "@astryxdesign/core/Grid";
import { Heading } from "@astryxdesign/core/Heading";
import { HStack } from "@astryxdesign/core/HStack";
import { Icon } from "@astryxdesign/core/Icon";
import { Link } from "@astryxdesign/core/Link";
import { Spinner } from "@astryxdesign/core/Spinner";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import { loadPlayer, parsePlayerId } from "@/src/data/load";
import type { Meta, PlayerDetail as PlayerDetailData, Role } from "@/src/data/types";
import { useJson } from "@/src/data/useJson";
import { BAND_NOTE, PLAYING_TIME_NOTE } from "@/src/site";
import { FieldingTable } from "./FieldingTable";
import { PlayerBand } from "./PlayerBand";
import { BatSeasonTable, PitSeasonTable, recentSeasons } from "./SeasonTable";
import { chartMetric, errorSummary, METRIC_LABEL, projectionCards, type RoleDetail } from "./text";
import { TrendChart } from "./TrendChart";

// ---------------------------------------------------------------------------
// 불러오기·오류 상태 — 모두 <main> 안에 안내와 예측표 링크를 둔다
// ---------------------------------------------------------------------------

/** 받는 동안 (Suspense 대체 화면으로도 쓴다) */
export function PlayerLoading() {
  return (
    <VStack paddingBlock={10} paddingInline={4} hAlign="center">
      <Spinner size="lg" label="선수 기록을 불러오는 중…" aria-label="선수 기록을 불러오는 중" />
    </VStack>
  );
}

function PlayerMessage({ title, description, icon }: { title: string; description: string; icon: typeof MagnifyingGlassIcon }) {
  return (
    <VStack hAlign="center">
      <VStack paddingBlock={10} paddingInline={4} width="100%" maxWidth={640}>
        <EmptyState
          headingLevel={1}
          title={title}
          description={description}
          icon={<Icon icon={icon} size="lg" color="secondary" />}
          actions={<Button label="예측표로 돌아가기" variant="primary" href="/" />}
        />
      </VStack>
    </VStack>
  );
}

// ---------------------------------------------------------------------------
// 본문 구역
// ---------------------------------------------------------------------------

function roleDetails(detail: PlayerDetailData): RoleDetail[] {
  const out: RoleDetail[] = [];
  if (detail.bat !== null) out.push({ role: "bat", detail: detail.bat });
  if (detail.pit !== null) out.push({ role: "pit", detail: detail.pit });
  return out;
}

/** 최근 팀: 보고 있는 역할의 마지막 기록 행, 없으면 다른 역할 */
function latestTeam(current: RoleDetail, all: RoleDetail[]): string | null {
  for (const rd of [current, ...all.filter((r) => r.role !== current.role)]) {
    const last = recentSeasons<{ season: number; team_name: string | null }>(rd.detail.history).at(-1);
    if (last?.team_name != null && last.team_name !== "") return last.team_name;
  }
  return null;
}

function ErrorNotes({ current }: { current: RoleDetail }) {
  return (
    <Grid columns={{ minWidth: 300, max: 2 }} gap={3}>
      <Card padding={5}>
        <HStack gap={3} vAlign="start">
          <Icon icon={InformationCircleIcon} color="info" />
          <VStack gap={2}>
            <Text as="h2" weight="bold">
              이 예측은 얼마나 빗나갈까
            </Text>
            <Text color="secondary" hasTabularNumbers>
              {errorSummary(current)}
            </Text>
            <Link href="/validation/">
              <HStack gap={1} vAlign="center">
                검증 자세히 보기
                <Icon icon={ArrowRightIcon} size="sm" color="inherit" />
              </HStack>
            </Link>
          </VStack>
        </HStack>
      </Card>
      <Card padding={5}>
        <HStack gap={3} vAlign="start">
          <Icon icon={InformationCircleIcon} color="accent" />
          <VStack gap={2}>
            <Text as="h2" weight="bold">
              출전량은 보수적 예상
            </Text>
            <Text color="secondary">{PLAYING_TIME_NOTE}</Text>
          </VStack>
        </HStack>
      </Card>
    </Grid>
  );
}

function TrendSection({ current, targetSeason }: { current: RoleDetail; targetSeason: number }) {
  const metric = chartMetric(current.role);
  const { detail } = current;
  const points = recentSeasons<{ season: number; ops?: number | null; era?: number | null }>(detail.history).map(
    (s) => ({ season: s.season, value: (metric === "ops" ? s.ops : s.era) ?? null }),
  );
  const projection = current.role === "bat" ? current.detail.projection.ops : current.detail.projection.era;
  const league = current.role === "bat" ? current.detail.league_baseline.ops : current.detail.league_baseline.era;
  const band = detail.group_error.band80[metric];
  return (
    <VStack as="section" gap={3} aria-labelledby="player-trend">
      <Heading level={2} id="player-trend">
        {`시즌별 ${METRIC_LABEL[metric]}`}
      </Heading>
      <Card padding={5}>
        <TrendChart
          key={current.role}
          metric={metric}
          points={points}
          targetSeason={targetSeason}
          projection={projection}
          band={band}
          league={league}
        />
      </Card>
      {band != null && (
        <Text type="supporting" color="secondary">
          {`${targetSeason} 점의 세로 막대: ${BAND_NOTE}`}
        </Text>
      )}
    </VStack>
  );
}

function SeasonSection({ current }: { current: RoleDetail }) {
  return (
    <VStack as="section" gap={3} aria-labelledby="player-seasons">
      <Heading level={2} id="player-seasons">
        최근 5시즌 기록
      </Heading>
      {current.role === "bat" ? (
        <BatSeasonTable history={current.detail.history} labelledBy="player-seasons" />
      ) : (
        <PitSeasonTable history={current.detail.history} labelledBy="player-seasons" />
      )}
    </VStack>
  );
}

function FieldingSection({ detail, meta }: { detail: PlayerDetailData; meta: Meta }) {
  const window = detail.bat?.window;
  const first = window?.input_first ?? Math.min(...meta.input_seasons);
  const last = window?.input_last ?? Math.max(...meta.input_seasons);
  const span = first === last ? String(first) : `${first}~${last}`;
  const position = detail.player.primary_position;
  let reason = "";
  if (detail.bat !== null && position === "DH") reason = " — 지명타자로만 뛰었습니다";
  else if (detail.bat !== null && position === "TWP") reason = " — 타자로는 지명타자로만 뛰었습니다";
  // 투수만 있는 선수: 수비 기록(OAA·FRV)은 야수 포지션만 모은다 — "수비수라면 …" 안내는 맞지 않는다
  const emptyText =
    detail.bat === null
      ? `${span} 수비 기록이 없습니다 — 수비 기록(OAA·FRV)은 야수 포지션만 모으고, 투수 수비는 모으지 않습니다.`
      : `${span} 수비 기록이 없습니다${reason}. 수비수라면 시즌별 포지션 아웃 수, OAA, FRV(포수는 프레이밍·블로킹·도루 저지)가 여기에 나옵니다. FRV와 OAA는 단위가 달라 더하지 않습니다.`;
  return (
    <VStack as="section" gap={3} aria-labelledby="player-fielding">
      <Heading level={2} id="player-fielding">
        수비 (과거 기록만)
      </Heading>
      {detail.fielding.length > 0 ? (
        <>
          <FieldingTable seasons={detail.fielding} labelledBy="player-fielding" />
          <Text type="supporting" color="secondary">
            포지션 옆 숫자는 그 포지션에서 수비한 아웃 수입니다. OAA는 아웃, FRV는 실점 단위라 더하지 않습니다.
          </Text>
        </>
      ) : (
        <Text color="secondary">{emptyText}</Text>
      )}
    </VStack>
  );
}

function PlayerView({ detail }: { detail: PlayerDetailData }) {
  const all = roleDetails(detail);
  // 투타 겸업은 타자가 기본 (SA-2). 역할은 주소에 담지 않는다
  const [role, setRole] = useState<Role>(all[0]?.role ?? "bat");
  const current = all.find((rd) => rd.role === role) ?? all[0];
  const targetSeason = detail.meta.target_season;

  if (current === undefined) {
    return (
      <PlayerMessage
        title={`${detail.player.name}의 ${targetSeason} 예측이 없습니다`}
        description="이 선수는 예측 대상이 아닙니다. 예측표에서 다른 선수를 골라 주세요."
        icon={MagnifyingGlassIcon}
      />
    );
  }

  return (
    <VStack gap={0}>
      <PlayerBand
        player={detail.player}
        targetSeason={targetSeason}
        current={current}
        roles={all.map((rd) => rd.role)}
        onRoleChange={setRole}
        team={latestTeam(current, all)}
        cards={projectionCards(current, targetSeason)}
      />
      <VStack hAlign="center">
        <VStack gap={8} paddingBlock={8} paddingInline={6} width="100%" maxWidth={1200}>
          <ErrorNotes current={current} />
          <TrendSection current={current} targetSeason={targetSeason} />
          <SeasonSection current={current} />
          <FieldingSection detail={detail} meta={detail.meta} />
        </VStack>
      </VStack>
    </VStack>
  );
}

// ---------------------------------------------------------------------------
// 주소 → 데이터 → 화면
// ---------------------------------------------------------------------------

/** useSearchParams를 쓰므로 page.tsx가 <Suspense>로 감싼다 (정적 export) */
export function PlayerDetail() {
  const params = useSearchParams();
  const id = parsePlayerId(params);
  const state = useJson(() => loadPlayer(id ?? 0), id === null ? null : String(id));

  if (id === null) {
    const hasId = params?.get("id") != null;
    return (
      <PlayerMessage
        title={hasId ? "선수 ID가 올바르지 않습니다" : "선수 ID가 없습니다"}
        description="주소의 ?id= 뒤에는 MLB 선수 번호(양의 정수)가 와야 합니다. 예측표에서 선수를 골라 주세요."
        icon={ExclamationTriangleIcon}
      />
    );
  }
  if (state.status === "loading") return <PlayerLoading />;
  if (state.status === "error") {
    return state.notFound ? (
      <PlayerMessage
        title="선수를 찾을 수 없습니다"
        description="이 선수 번호의 예측이 없습니다. 예측 대상은 최근 5시즌에 MLB에서 뛴 선수입니다."
        icon={MagnifyingGlassIcon}
      />
    ) : (
      <PlayerMessage
        title="선수 기록을 불러오지 못했습니다"
        description="네트워크나 서버 문제일 수 있습니다. 잠시 뒤 다시 시도해 주세요."
        icon={ExclamationTriangleIcon}
      />
    );
  }
  return <PlayerView key={id} detail={state.data} />;
}
