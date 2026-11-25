// 소개 (SA-5) — 방법·검증 절차·데이터·알아 두실 점·고지. 정적 글이라 데이터를 받지 않는다.
// 틀은 Astryx documentation-technical 템플릿(Layout + LayoutContent, 960 열, padding 8, gap 8)에서 왔다.
// 서버 컴포넌트로 두어 화면 제목(metadata)을 내보낸다 — 레이아웃 제목 틀이 "소개 · 사이트 이름"으로 만든다.
import type { Metadata } from "next";
import Image from "next/image";
import { Card } from "@astryxdesign/core/Card";
import { Divider } from "@astryxdesign/core/Divider";
import { Grid } from "@astryxdesign/core/Grid";
import { Heading } from "@astryxdesign/core/Heading";
import { HStack } from "@astryxdesign/core/HStack";
import { Layout, LayoutContent } from "@astryxdesign/core/Layout";
import { Link } from "@astryxdesign/core/Link";
import { List, ListItem } from "@astryxdesign/core/List";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import { photosEnabled } from "@/src/photo";
import { REPO_URL } from "@/src/site";
import { SiteFooter } from "../components/SiteFooter";
import { SiteHeader } from "../components/SiteHeader";

export const metadata: Metadata = { title: "소개" };

// 방법 4단계 (SA-5 문구 그대로 — 가중치는 config/params_v1.json, 보정 지표는 SPEC F9와 같다)
const STEPS: ReadonlyArray<{ title: string; body: string }> = [
  {
    title: "최근 시즌일수록 크게",
    body: "최대 5시즌 기록을 합치되, 직전 시즌에 가장 큰 무게를 줍니다(8·4·2·1·0.5).",
  },
  {
    title: "리그 평균 쪽으로 당기기",
    body: "기록이 적을수록 리그 평균에 가깝게 예측합니다. 한두 시즌 반짝한 기록을 그대로 믿지 않기 위해서입니다.",
  },
  {
    title: "출전량은 따로",
    body: "타석·이닝은 직전 출전량, 공백, 선발 비중으로 따로 예측합니다. 작년에 뛰지 않은 선수는 비슷한 처지 선수들의 실제 값으로 봅니다.",
  },
  {
    title: "나이·Statcast로 다듬기",
    body: "나이, 삼진율, 타구 질(배럴·타구 속도·xwOBA), 투수의 헛스윙률로 조금 보정합니다. 과거 검증에서 오차가 줄어든 지표(HR·OPS·ERA·K−BB%)에만 씁니다.",
  },
];

const DATA_ITEMS = [
  "MLB Stats API — 시즌 기록, 선수 정보",
  "Baseball Savant — Statcast 지표, OAA, Fielding Run Value",
  "2019~2026 정규 시즌 (2020년은 60경기라 출전량만 162경기 기준으로 늘려 봅니다)",
  "2026 시즌 종료 기준으로 한 번 만들었고, 다음 갱신은 사람이 직접 돌립니다",
];

// 알아 두실 점 — 출전량 보수적(D30) · 범위가 넓음(D29) · 신인 없음 · 팀 이동 미반영
const CAVEATS = [
  "출전량(PA·IP)과 HR은 부상·결장 위험을 평균에 넣은 보수적 예상입니다. 주전 선수는 평소보다 낮게 나오는 편입니다.",
  "± 범위는 넓습니다. OPS는 ±.10 안팎, ERA는 ±1.4 안팎 — 한 시즌 성적은 그만큼 흔들립니다.",
  "MLB 기록이 없는 선수(신인, 마이너리그·해외 리그 출신)는 예측하지 않습니다.",
  "팀 이동·구장·계약은 반영하지 않습니다.",
];

export default function AboutPage() {
  // 사진을 끄면(NEXT_PUBLIC_PLAYER_PHOTOS=off) 고지에서도 사진 이야기를 뺀다 (D27)
  const rightsSubject = photosEnabled() ? "MLB 자료와 선수 사진의" : "MLB 자료의";

  return (
    <VStack gap={0} minHeight="100dvh">
      <SiteHeader active="about" />
      <Layout
        height="auto"
        contentWidth={960}
        content={
          <LayoutContent role="main" isScrollable={false} padding={8}>
            <VStack gap={8}>
              <VStack gap={3}>
                <Text type="label" color="accent" weight="bold">
                  ABOUT · HOW IT WORKS
                </Text>
                <Heading level={1} type="display-2">
                  이 예측은 어떻게 만들었나
                </Heading>
                <Text as="p" type="large" color="secondary">
                  최근 5시즌(2022~2026)에 MLB 기록이 있는 선수 전원의 2027 성적을 과거 기록만으로 계산했습니다.
                  팬이 재미로 만든 예측이며 구단·리그와 관계가 없습니다.
                </Text>
              </VStack>

              <VStack as="section" gap={4} aria-labelledby="about-method">
                <Heading id="about-method" level={2}>
                  방법
                </Heading>
                {/* 섹션 이미지는 장식이다 (alt="", D19) */}
                <Image
                  src="/images/section-method.jpg"
                  alt=""
                  width={1800}
                  height={600}
                  unoptimized
                  className="h-48 w-full rounded-lg object-cover"
                />
                <Grid columns={{ minWidth: 320, max: 2 }} gap={4}>
                  {STEPS.map((step, i) => (
                    <Card key={step.title}>
                      <VStack gap={2}>
                        <HStack gap={2} vAlign="center">
                          {/* 단계 번호 — 장식 (제목 순서가 단계 순서다) */}
                          <HStack
                            as="span"
                            aria-hidden="true"
                            hAlign="center"
                            vAlign="center"
                            className="size-7 shrink-0 rounded-full bg-accent-bg text-on-accent"
                          >
                            <Text weight="bold" size="sm" color="inherit" hasTabularNumbers>
                              {String(i + 1)}
                            </Text>
                          </HStack>
                          <Heading level={3}>{step.title}</Heading>
                        </HStack>
                        <Text as="p" color="secondary">
                          {step.body}
                        </Text>
                      </VStack>
                    </Card>
                  ))}
                </Grid>
              </VStack>

              <Divider />

              <VStack as="section" gap={4} aria-labelledby="about-validation">
                <Heading id="about-validation" level={2}>
                  어떻게 검증했나
                </Heading>
                {/* ListItem은 문자열(조각으로 감싼 문자열 포함) label을 한 줄로 자른다 — 문장은 Text로 넘겨 줄바꿈되게 한다 */}
                <List density="compact" listStyle="decimal">
                  <ListItem label={<Text>가중치·회귀 강도와 어떤 보정을 쓸지는 2024·2025 시즌을 맞혀 보며 정했습니다.</Text>} />
                  <ListItem label={<Text>그 설정을 고정하고, 2026 시즌을 한 번만 시험했습니다.</Text>} />
                  <ListItem
                    label={
                      <Text>
                        결함을 고친 뒤 2026 시험을 한 번 더 돌렸고, 두 결과를 모두 공개합니다.{" "}
                        <Link href="/validation/">검증 결과 보기</Link>
                      </Text>
                    }
                  />
                </List>
              </VStack>

              <Divider />

              <VStack as="section" gap={4} aria-labelledby="about-data">
                <Heading id="about-data" level={2}>
                  데이터
                </Heading>
                <List density="compact" listStyle="disc">
                  {DATA_ITEMS.map((item) => (
                    <ListItem key={item} label={<Text>{item}</Text>} />
                  ))}
                </List>
              </VStack>

              <Divider />

              <VStack as="section" gap={4} aria-labelledby="about-caveats">
                <Heading id="about-caveats" level={2}>
                  알아 두실 점
                </Heading>
                <List density="compact" listStyle="disc">
                  {CAVEATS.map((item) => (
                    <ListItem key={item} label={<Text>{item}</Text>} />
                  ))}
                </List>
              </VStack>

              <Card variant="muted">
                <VStack as="section" gap={2} aria-labelledby="about-notice">
                  <Heading id="about-notice" level={2}>
                    고지
                  </Heading>
                  <Text as="p" color="secondary">
                    재미로 보는 예측입니다. 실제 성적을 보장하지 않으며 베팅·판타지 조언이 아닙니다. {rightsSubject} 권리는
                    MLB Advanced Media, L.P.에 있습니다. 소스 코드는{" "}
                    <Link href={REPO_URL} isExternalLink newTabLabel="(새 창에서 열림)">
                      GitHub
                    </Link>
                    에 공개되어 있습니다.
                  </Text>
                </VStack>
              </Card>
            </VStack>
          </LayoutContent>
        }
      />
      <SiteFooter />
    </VStack>
  );
}
