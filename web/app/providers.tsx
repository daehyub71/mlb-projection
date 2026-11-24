"use client";

import Link from "next/link";
import { Theme } from "@astryxdesign/core/theme";
import { LinkProvider } from "@astryxdesign/core/Link";
import { defaultThemeSlug, themeModes, themes } from "@/src/astryx-themes";

// Astryx 테마 공급자는 클라이언트 컴포넌트다 (SPEC E14). 사이트 전용 테마 ballpark, 다크 전용 (DESIGN DS5)
export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <Theme theme={themes[defaultThemeSlug]} mode={themeModes[defaultThemeSlug] ?? "system"}>
      <LinkProvider component={Link}>{children}</LinkProvider>
    </Theme>
  );
}
