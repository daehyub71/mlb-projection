// 선수 상세 (F14, 시안 SA-2) — 주소 /player/?id=660271 한 페이지가 players/{id}.json을 읽는다 (DS1)
// detail-page 템플릿의 Layout 틀(머리글·본문·바닥글)에서 시작했다. 정적 export라 ?id=는 브라우저에서 읽고,
// useSearchParams를 쓰는 본문은 <Suspense>로 감싼다 (빌드 때는 불러오는 중 화면이 정적 HTML이 된다).
import { Suspense } from "react";
import type { Metadata } from "next";
import { Layout, LayoutContent } from "@astryxdesign/core/Layout";
import { SiteFooter } from "../components/SiteFooter";
import { SiteHeader } from "../components/SiteHeader";
import { PlayerDetail, PlayerLoading } from "./PlayerDetail";

export const metadata: Metadata = {
  title: "선수 상세",
};

export default function PlayerPage() {
  return (
    <Layout
      height="auto"
      header={<SiteHeader active="projections" />}
      content={
        <LayoutContent role="main" isScrollable={false} padding={0}>
          <Suspense fallback={<PlayerLoading />}>
            <PlayerDetail />
          </Suspense>
        </LayoutContent>
      }
      footer={<SiteFooter />}
    />
  );
}
