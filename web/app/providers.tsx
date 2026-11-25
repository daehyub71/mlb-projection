"use client";

import Link from "next/link";
import { Theme } from "@astryxdesign/core/theme";
import { LinkProvider } from "@astryxdesign/core/Link";
import { InternationalizationProvider } from "@astryxdesign/core/i18n";
import koKR from "@astryxdesign/core/locales/ko-KR.generated.js";
import { defaultThemeSlug, themeModes, themes } from "@/src/astryx-themes";

// Astryx 테마 공급자는 클라이언트 컴포넌트다 (SPEC E14). 사이트 전용 테마 ballpark, 다크 전용 (DESIGN DS5)
export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <Theme theme={themes[defaultThemeSlug]} mode={themeModes[defaultThemeSlug] ?? "system"}>
      {/* Astryx 기본 문구(쪽 이동·정렬·차트 축 등)를 한국어로 — 모든 화면 공통 */}
      <InternationalizationProvider locale="ko-KR" messages={{ "ko-KR": koKR }}>
        <LinkProvider component={Link}>{children}</LinkProvider>
      </InternationalizationProvider>
    </Theme>
  );
}
