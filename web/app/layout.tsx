import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { SITE_DESCRIPTION, SITE_NAME, SITE_TITLE } from "@/src/site";

// 사이트 메타데이터 (SPEC F17). 하위 화면은 title만 주면 "{화면} · 사이트 이름"이 된다.
// 공유 이미지는 app/opengraph-image.tsx가 만든다(og-bg.jpg 위에 제목, D19) — Next가 og:image를 붙인다.
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
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
