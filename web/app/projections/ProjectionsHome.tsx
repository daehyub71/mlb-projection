"use client";

// 예측표 첫 화면 (F13, DESIGN SA-1 데스크톱 · SA-3 모바일, DS2~DS5)
// 틀은 Astryx table-page 템플릿(LayoutContent padding 0 + 안쪽 VStack이 여백을 갖는다, 검색·범위 토글 → 표 → 빈 상태)에서 왔다.
// 데이터는 정적 export라 브라우저에서 받는다: 타자 표는 처음에, 투수 표는 처음 투수 탭을 누를 때 한 번만.
// 보기 상태(탭·검색·필터·정렬·쪽)는 주소 해시에 담는다 — 선수 상세에서 뒤로 오면 그대로 돌아온다 (DESIGN §1-2, viewState.ts).
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { useMediaQuery } from "@astryxdesign/core/hooks";
import { HStack } from "@astryxdesign/core/HStack";
import { InternationalizationProvider } from "@astryxdesign/core/i18n";
import { Layout, LayoutContent } from "@astryxdesign/core/Layout";
import { Link } from "@astryxdesign/core/Link";
import { Pagination } from "@astryxdesign/core/Pagination";
import { Selector } from "@astryxdesign/core/Selector";
import { Spinner } from "@astryxdesign/core/Spinner";
import { Switch } from "@astryxdesign/core/Switch";
import { Tab, TabList } from "@astryxdesign/core/TabList";
import { Table, useTableSortable } from "@astryxdesign/core/Table";
import { Text } from "@astryxdesign/core/Text";
import { TextInput } from "@astryxdesign/core/TextInput";
import { VStack } from "@astryxdesign/core/VStack";
import koKR from "@astryxdesign/core/locales/ko-KR.generated.js";
import { loadBatters, loadPitchers } from "@/src/data/load";
import type { Meta, Role } from "@/src/data/types";
import { useJson, type JsonState } from "@/src/data/useJson";
import { fmtCount } from "@/src/format";
import { BAND_NOTE, CAREER_LABEL, PAGE_SIZE, PLAYING_TIME_NOTE } from "@/src/site";
import { SiteFooter } from "../components/SiteFooter";
import { SiteHeader } from "../components/SiteHeader";
import { batColumns, pitColumns, type SeasonLabels } from "./columns";
import { Hero } from "./Hero";
import { BatList, PitList } from "./MobileList";
import {
  filterRows,
  filterStarters,
  PIT_MIN_IP,
  nextSort,
  pageCount,
  sortRows,
  toBatViews,
  toPitViews,
  type BatSortKey,
  type BatView,
  type CareerFilter,
  type PitSortKey,
  type PitView,
  type SortEntry,
} from "./rows";
import { DEFAULT_VIEW, decodeView, encodeView, isViewHash, type ViewState } from "./viewState";

/** 표 영역의 앵커 — 히어로의 "예측표 보기"가 이리로 온다 */
const TABLE_ANCHOR = "projections";
const PANEL_ID = "projections-panel";

/** SA-3 모바일 카드 목록으로 바꾸는 너비 (Tailwind md 미만과 같은 경계) */
const NARROW_QUERY = "(max-width: 47.99rem)";

/** 표 위 고지 한 줄 (D16, SA-1 합의 문구) — 바닥글의 짧은 고지와 별도로 둔다 */
const NOTICE = "재미로 보는 예측입니다. 실제 성적을 보장하지 않으며 베팅 조언이 아닙니다.";

const ROLE_LABEL: Record<Role, string> = { bat: "타자", pit: "투수" };

const CAREER_OPTIONS: { value: CareerFilter; label: string }[] = [
  { value: "all", label: "전체" },
  { value: "seasons_1_2", label: CAREER_LABEL.seasons_1_2 },
  { value: "seasons_3_5", label: CAREER_LABEL.seasons_3_5 },
];

/** 모바일 정렬 선택 (SA-3) — 머리를 누를 수 없어서 따로 둔다. 값은 "열:방향" */
const MOBILE_SORTS: Record<Role, { value: string; label: string }[]> = {
  bat: [
    { value: "ops:descending", label: "OPS 높은 순" },
    { value: "hr:descending", label: "HR 많은 순" },
    { value: "pa:descending", label: "PA 많은 순" },
  ],
  pit: [
    { value: "era:ascending", label: "ERA 낮은 순" },
    { value: "kbb:descending", label: "K−BB% 높은 순" },
    { value: "ip:descending", label: "IP 많은 순" },
  ],
};

const subscribeNothing = () => () => {};

/**
 * 하이드레이션이 끝났는가 — 서버(정적 HTML)와 하이드레이션 때는 false, 그 뒤 true.
 * 해시는 서버에 없으므로, 하이드레이션은 기본 보기로 하고 바로 다음 그리기에서 해시를 읽는다(불일치 없음).
 */
function useIsClient(): boolean {
  return useSyncExternalStore(
    subscribeNothing,
    () => true,
    () => false,
  );
}

/** 보기 상태를 주소 해시에 적는다 — 기록을 늘리지 않게 replaceState. "#projections" 같은 앵커는 지우지 않는다 */
function writeViewHash(encoded: string) {
  const { pathname, search, hash } = window.location;
  if (hash.replace(/^#/, "") === encoded) return;
  if (encoded === "" && !isViewHash(hash)) return;
  window.history.replaceState(null, "", `${pathname}${search}${encoded === "" ? "" : `#${encoded}`}`);
}

function seasonLabels(meta: Meta): SeasonLabels {
  const seasons = meta.input_seasons;
  return { target: meta.target_season, prev: seasons.length > 0 ? Math.max(...seasons) : meta.target_season - 1 };
}

function sortOptionValue(sort: readonly SortEntry[]): string | undefined {
  const s = sort[0];
  return s === undefined ? undefined : `${s.sortKey}:${s.direction}`;
}

function parseSortOption<K extends string>(value: string | null): SortEntry<K>[] | null {
  if (value === null) return null;
  const [key, direction] = value.split(":");
  if (key === undefined || (direction !== "ascending" && direction !== "descending")) return null;
  return [{ sortKey: key as K, direction }];
}

/**
 * 받는 중 — 글자와 aria-busy로 알린다. Spinner 자체 알림은 끈다: Spinner는 문구를 공용 알림 영역에 남겨
 * 다 받은 뒤에도 "불러오는 중"이 화면 트리에 남는다. 알림은 이 영역(role=status)이 맡는다.
 */
function LoadingPanel({ role }: { role: Role }) {
  return (
    <VStack role="status" padding={10} gap={3} hAlign="center" aria-busy="true">
      <Spinner size="lg" aria-hidden />
      <Text color="secondary">{`${ROLE_LABEL[role]} 예측표를 불러오는 중…`}</Text>
    </VStack>
  );
}

/** 받지 못함 — 다시 받기 단추 */
function ErrorPanel({ role, notFound, onRetry }: { role: Role; notFound: boolean; onRetry: () => void }) {
  return (
    <Banner
      status="error"
      title={`${ROLE_LABEL[role]} 예측표를 불러오지 못했습니다`}
      description={notFound ? "예측표 파일을 찾지 못했습니다." : "네트워크 상태를 확인한 뒤 다시 시도해 주세요."}
      endContent={<Button label="다시 시도" variant="secondary" onClick={onRetry} />}
    />
  );
}

export function ProjectionsHome() {
  const isClient = useIsClient();
  // 사용자가 처음 바꾸기 전에는 주소 해시(뒤로 가기로 돌아온 보기)를, 없으면 첫 화면 기본값(DS2~DS4)을 쓴다
  const [touched, setTouched] = useState<ViewState | null>(null);
  const arrivedHash = isClient && touched === null ? window.location.hash : "";
  // 같은 해시는 한 번만 푼다 — 정렬 배열이 그릴 때마다 새로 생겨 표 정렬을 다시 하지 않게
  const arrived = useMemo(() => decodeView(arrivedHash), [arrivedHash]);
  const view: ViewState = touched ?? arrived ?? DEFAULT_VIEW;
  const { role, query, prevOnly, startersOnly, career, page, batSort, pitSort } = view;
  const [pitRequested, setPitRequested] = useState(false);
  const [attempts, setAttempts] = useState<Record<Role, number>>({ bat: 0, pit: 0 });
  const isNarrow = useMediaQuery(NARROW_QUERY);

  /** 보기 상태를 바꾼다 — 검색·필터·정렬·탭을 바꿀 때는 page: 1을 같이 넘긴다 (DS4) */
  const update = (patch: Partial<ViewState>) => setTouched((prev) => ({ ...(prev ?? view), ...patch }));

  // 바꾼 보기만 주소에 적는다 — 처음 들어온 주소(#projections 앵커 포함)는 건드리지 않는다
  useEffect(() => {
    if (touched !== null) writeViewHash(encodeView(touched));
  }, [touched]);

  const bat = useJson(loadBatters, `bat:${attempts.bat}`);
  // 투수 표는 한 번 요청하면 key를 계속 유지한다 — 타자 탭으로 돌아갔다 와도 다시 받지 않는다.
  // 주소 해시로 투수 탭이 열려 돌아온 경우도 요청한 것으로 본다
  const pit = useJson(loadPitchers, pitRequested || role === "pit" ? `pit:${attempts.pit}` : null);

  const batViews = useMemo(() => (bat.status === "ok" ? toBatViews(bat.data) : []), [bat]);
  const pitViews = useMemo(() => (pit.status === "ok" ? toPitViews(pit.data) : []), [pit]);

  const filter = { query, prevOnly, career };
  const batVisible = useMemo(
    () => sortRows(filterRows(batViews, { query, prevOnly, career }), batSort),
    [batViews, query, prevOnly, career, batSort],
  );
  const pitVisible = useMemo(
    () => sortRows(filterRows(filterStarters(pitViews, startersOnly), { query, prevOnly, career }), pitSort),
    [pitViews, startersOnly, query, prevOnly, career, pitSort],
  );

  const batSortPlugin = useTableSortable<BatView, BatSortKey>({
    sort: batSort,
    onSortChange: (next) => update({ batSort: nextSort(batSort, next), page: 1 }),
    allowUnsortedState: false,
  });
  const pitSortPlugin = useTableSortable<PitView, PitSortKey>({
    sort: pitSort,
    onSortChange: (next) => update({ pitSort: nextSort(pitSort, next), page: 1 }),
    allowUnsortedState: false,
  });

  const chooseRole = (next: string) => {
    if (next !== "bat" && next !== "pit") return;
    // 투수 탭을 한 번이라도 열었으면(해시로 열린 경우 포함) 투수 표를 계속 들고 있는다
    if (next === "pit" || role === "pit") setPitRequested(true);
    update({ role: next, page: 1 });
  };

  const retry = (r: Role) => setAttempts((a) => ({ ...a, [r]: a[r] + 1 }));

  const state: JsonState<unknown> = role === "bat" ? bat : pit;
  const visible: (BatView | PitView)[] = role === "bat" ? batVisible : pitVisible;
  const allViews: (BatView | PitView)[] = role === "bat" ? batViews : pitViews;
  const pages = pageCount(visible.length, PAGE_SIZE);
  const safePage = Math.min(page, pages);
  const start = (safePage - 1) * PAGE_SIZE;
  // 좁은 화면은 쪽 대신 "50명 더 보기" — page 묶음까지 이어서 보인다
  const shown = safePage * PAGE_SIZE;

  // 출전 필터 때문에 비었으면, 끄면 몇 명이 나오는지 알려 준다
  const hiddenByPrev =
    prevOnly && visible.length === 0 ? filterRows(allViews, { ...filter, prevOnly: false }).length : 0;

  const clearFilters = () => update({ query: "", prevOnly: true, startersOnly: true, career: "all", page: 1 });

  const goToPage = (next: number) => {
    update({ page: next });
    // 표 맨 위로 — 아래쪽 쪽 단추를 누른 뒤 새 쪽의 첫 행부터 읽도록
    document.getElementById(TABLE_ANCHOR)?.scrollIntoView?.({ block: "start" });
  };

  function renderPanel() {
    if (state.status === "loading") return <LoadingPanel role={role} />;
    if (state.status === "error") {
      return <ErrorPanel role={role} notFound={state.notFound} onRetry={() => retry(role)} />;
    }
    if (visible.length === 0) {
      return (
        <EmptyState
          headingLevel={2}
          title="조건에 맞는 선수가 없습니다"
          description={
            hiddenByPrev > 0
              ? `2026 출전 선수만 보고 있습니다. 끄면 ${fmtCount(hiddenByPrev)}명이 더 있습니다.`
              : "검색어나 사용 시즌 필터를 바꿔 보세요."
          }
          actions={
            hiddenByPrev > 0 ? (
              <Button
                label="2026에 뛰지 않은 선수도 보기"
                variant="secondary"
                onClick={() => update({ prevOnly: false, page: 1 })}
              />
            ) : (
              <Button label="필터 지우기" variant="secondary" onClick={clearFilters} />
            )
          }
        />
      );
    }
    const label = `${ROLE_LABEL[role]} 2027 예측`;
    if (role === "bat" && bat.status === "ok") {
      const seasons = seasonLabels(bat.data.meta);
      if (isNarrow) return <BatList rows={batVisible.slice(0, shown)} seasons={seasons} label={label} />;
      return (
        <Table<BatView>
          aria-label={label}
          data={batVisible.slice(start, start + PAGE_SIZE)}
          columns={batColumns(seasons, bat.data.league_baseline.ops ?? null)}
          idKey="id"
          density="balanced"
          hasHover
          plugins={{ sort: batSortPlugin }}
        />
      );
    }
    if (role === "pit" && pit.status === "ok") {
      const seasons = seasonLabels(pit.data.meta);
      if (isNarrow) return <PitList rows={pitVisible.slice(0, shown)} seasons={seasons} label={label} />;
      return (
        <Table<PitView>
          aria-label={label}
          data={pitVisible.slice(start, start + PAGE_SIZE)}
          columns={pitColumns(seasons)}
          idKey="id"
          density="balanced"
          hasHover
          plugins={{ sort: pitSortPlugin }}
        />
      );
    }
    return null;
  }

  const hasRows = state.status === "ok" && visible.length > 0;
  const currentSort = role === "bat" ? batSort : pitSort;

  return (
    <InternationalizationProvider locale="ko-KR" messages={{ "ko-KR": koKR }}>
      <VStack gap={0} minHeight="100dvh">
        <SiteHeader active="projections" />
        <Layout
          height="auto"
          content={
            // padding 0은 템플릿 그대로 — 표가 컨테이너 여백만큼 밖으로 번지지 않게, 여백은 안쪽 VStack이 갖는다
            <LayoutContent role="main" isScrollable={false} padding={0}>
              <VStack gap={0}>
                <Hero
                  tableAnchor={TABLE_ANCHOR}
                  counts={{
                    batters: bat.status === "ok" ? bat.data.rows.length : null,
                    // 투수 표는 지연 로딩이라 수는 meta.counts에서 (SA-1)
                    pitchers: bat.status === "ok" ? bat.data.meta.counts.pit : pit.status === "ok" ? pit.data.rows.length : null,
                    inputSeasons: bat.status === "ok" ? bat.data.meta.input_seasons : null,
                  }}
                />
                <VStack
                  as="section"
                  id={TABLE_ANCHOR}
                  aria-label="2027 예측표"
                  width="100%"
                  maxWidth={1200}
                  gap={5}
                  paddingInline={6}
                  paddingBlock={8}
                  // 가운데 정렬 + 붙어 있는 머리글(sticky) 아래로 앵커가 숨지 않게
                  className="mx-auto scroll-mt-16"
                >
                  <Text as="p" color="secondary">
                    {NOTICE}
                  </Text>

                  {/* 탭이 먼저, 그다음 검색·필터 — 탭이 무엇을 찾을지 정한다 */}
                  <HStack gap={4} hAlign="between" vAlign="end" wrap="wrap">
                    <TabList role="tablist" aria-label="역할" value={role} onChange={chooseRole} size="lg" hasDivider>
                      <Tab value="bat" label="타자" panelId={PANEL_ID} />
                      <Tab value="pit" label="투수" panelId={PANEL_ID} />
                    </TabList>
                    <HStack gap={3} vAlign="center" wrap="wrap">
                      <VStack className="w-full md:w-56">
                        <TextInput
                          type="search"
                          label="선수 이름 검색"
                          isLabelHidden
                          placeholder="선수 이름 검색"
                          startIcon="search"
                          hasClear
                          value={query}
                          onChange={(value) => update({ query: value, page: 1 })}
                        />
                      </VStack>
                      <Switch
                        label="2026 출전 선수만"
                        value={prevOnly}
                        onChange={(checked) => update({ prevOnly: checked, page: 1 })}
                      />
                      {role === "pit" && (
                        <Switch
                          label={`예상 ${PIT_MIN_IP}이닝 이상`}
                          value={startersOnly}
                          onChange={(checked) => update({ startersOnly: checked, page: 1 })}
                        />
                      )}
                      {/* 사용 시즌 그룹은 넓은 화면만 (SA-1), 좁은 화면은 정렬 선택 (SA-3) */}
                      <VStack className="hidden md:flex">
                        <Selector
                          label="사용 시즌"
                          isLabelHidden
                          options={CAREER_OPTIONS}
                          value={career}
                          renderValue={(option) => `사용 시즌: ${String(option.label)}`}
                          onChange={(value) => update({ career: (value as CareerFilter | null) ?? "all", page: 1 })}
                        />
                      </VStack>
                      <VStack className="md:hidden">
                        <Selector
                          label="정렬"
                          isLabelHidden
                          placeholder="정렬"
                          options={MOBILE_SORTS[role]}
                          value={sortOptionValue(currentSort)}
                          onChange={(value) => {
                            if (role === "bat") {
                              const next = parseSortOption<BatSortKey>(value);
                              if (next !== null) update({ batSort: next, page: 1 });
                            } else {
                              const next = parseSortOption<PitSortKey>(value);
                              if (next !== null) update({ pitSort: next, page: 1 });
                            }
                          }}
                        />
                      </VStack>
                    </HStack>
                  </HStack>

                  <VStack role="tabpanel" id={PANEL_ID} aria-label={`${ROLE_LABEL[role]} 예측`} gap={4}>
                    {renderPanel()}
                  </VStack>

                  {/* 표 아래 설명 (D29·D30) */}
                  <VStack gap={1}>
                    <Text as="p" type="supporting" color="secondary">
                      {BAND_NOTE}{" "}
                      <Link href="/validation/" type="supporting">
                        검증 보기
                      </Link>
                    </Text>
                    <Text as="p" type="supporting" color="secondary">
                      {PLAYING_TIME_NOTE}
                    </Text>
                  </VStack>

                  {hasRows && !isNarrow && pages > 1 && (
                    <HStack hAlign="center">
                      <Pagination
                        label="페이지"
                        page={safePage}
                        onChange={goToPage}
                        totalItems={visible.length}
                        pageSize={PAGE_SIZE}
                      />
                    </HStack>
                  )}
                  {hasRows && isNarrow && shown < visible.length && (
                    <Button
                      label={`${fmtCount(PAGE_SIZE)}명 더 보기`}
                      variant="secondary"
                      width="100%"
                      onClick={() => update({ page: safePage + 1 })}
                    />
                  )}
                </VStack>
              </VStack>
            </LayoutContent>
          }
        />
        <SiteFooter />
      </VStack>
    </InternationalizationProvider>
  );
}
