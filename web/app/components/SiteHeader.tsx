// 모든 화면 머리글 — 사이트 표시와 메뉴 (DESIGN v1.0 §6, SA 공통). 이름·props는 계약
// 화면 위에 붙는다(sticky). 메뉴는 머리글의 마지막에 둔다 — 머리글 안에는 제목(heading)을 넣지 않는다.
import { HStack } from "@astryxdesign/core/HStack";
import { Link } from "@astryxdesign/core/Link";
import { Text } from "@astryxdesign/core/Text";
import { NAV, SITE_NAME, type NavKey } from "@/src/site";

export function SiteHeader({ active }: { active: NavKey | null }) {
  return (
    <HStack
      as="header"
      gap={4}
      paddingInline={6}
      paddingBlock={3}
      hAlign="between"
      vAlign="center"
      wrap="wrap"
      className="sticky top-0 z-10 border-b border-border bg-body"
    >
      <Link href="/" label={SITE_NAME} color="primary">
        <HStack as="span" gap={2} vAlign="center">
          {/* 사이트 마크 — 장식 (이름은 링크 label이 읽는다) */}
          <HStack
            as="span"
            aria-hidden="true"
            hAlign="center"
            vAlign="center"
            className="size-7 shrink-0 rounded-md bg-accent-bg text-on-accent"
          >
            <Text weight="bold" size="sm" color="inherit" hasTabularNumbers>
              27
            </Text>
          </HStack>
          <Text weight="bold" size="lg">
            {SITE_NAME}
          </Text>
        </HStack>
      </Link>
      <HStack as="nav" gap={1} vAlign="center" aria-label="주요 메뉴">
        {NAV.map((item) => {
          const isCurrent = item.key === active;
          return (
            <Link
              key={item.key}
              href={item.href}
              aria-current={isCurrent ? "page" : undefined}
              color={isCurrent ? "primary" : "secondary"}
              // 현재 화면은 색만이 아니라 채운 배경·굵은 글자로도 구분한다 (N10)
              className={isCurrent ? "rounded-md bg-accent-muted px-3 py-2" : "rounded-md px-3 py-2"}
            >
              <Text weight={isCurrent ? "bold" : "medium"} color="inherit">
                {item.label}
              </Text>
            </Link>
          );
        })}
      </HStack>
    </HStack>
  );
}
