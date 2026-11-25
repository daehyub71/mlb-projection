import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { SITE_DESCRIPTION, SITE_NAME, SITE_TITLE } from "@/src/site";

// 사이트 메타데이터 (SPEC F17). 하위 화면은 title만 주면 "{화면} · 사이트 이름"이 된다.
// 공유 이미지는 app/opengraph-image.jpg(1200×630, 약 130KB) — Next가 og:image를 붙인다. og-bg.jpg 위에 제목을 코드로 얹어
// 만든 PNG를 JPEG로 바꾼 것이다(D19·N9). 제목을 바꾸면 커밋 17466b4의 app/opengraph-image.tsx로 다시 만든다(DESIGN §5).
export const metadata: Metadata = {
  title: { default: SITE_TITLE, template: `%s · ${SITE_NAME}` },
  description: SITE_DESCRIPTION,
  openGraph: {
    type: "website",
    locale: "ko_KR",
    siteName: SITE_NAME,
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      {/* 페이지 바탕은 테마 토큰으로 — 브라우저 기본 다크 캔버스에 기대지 않는다 (N10) */}
      <body className="bg-body text-primary">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
