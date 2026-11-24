// 모든 화면 머리글 — 사이트 이름과 메뉴 (DESIGN v1.0 §6). 모양 다듬기는 레인 C, 이름·props는 계약
import { HStack } from "@astryxdesign/core/HStack";
import { Link } from "@astryxdesign/core/Link";
import { Text } from "@astryxdesign/core/Text";
import { NAV, SITE_NAME, type NavKey } from "@/src/site";

export function SiteHeader({ active }: { active: NavKey | null }) {
  return (
    <HStack as="header" gap={4} padding={4} hAlign="between" vAlign="center" wrap="wrap">
      <Link href="/" label={SITE_NAME}>
        <Text weight="bold" size="lg">
          {SITE_NAME}
        </Text>
      </Link>
      <HStack as="nav" gap={4} vAlign="center" aria-label="주요 메뉴">
        {NAV.map((item) => (
          <Link key={item.key} href={item.href} aria-current={item.key === active ? "page" : undefined}>
            <Text weight={item.key === active ? "bold" : "normal"} color={item.key === active ? "accent" : "secondary"}>
              {item.label}
            </Text>
          </Link>
        ))}
      </HStack>
    </HStack>
  );
}
