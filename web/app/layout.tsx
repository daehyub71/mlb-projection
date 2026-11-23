import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "재미로 보는 2027 MLB 선수 성적 예측",
  description: "최근 5시즌 기록으로 만든 2027 MLB 타자·투수 성적 예측 (재미로 보는 예측)",
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
