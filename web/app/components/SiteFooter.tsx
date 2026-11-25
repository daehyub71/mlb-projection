// 모든 화면 바닥글 — 고지·출처·사진 저작권 (SPEC D16·D27, F16). 문구는 계약, 모양은 SA 공통
// 다른 레인 테스트가 실제 <footer> 요소와 "출처 · 사진 저작권" 한 줄(한 요소)을 찾는다 — 둘 다 유지한다.
import { HStack } from "@astryxdesign/core/HStack";
import { Link } from "@astryxdesign/core/Link";
import { Text } from "@astryxdesign/core/Text";
import { PHOTO_CREDIT, photosEnabled } from "@/src/photo";
import { DISCLAIMER_SHORT, REPO_URL, SOURCES_LINE } from "@/src/site";

export function SiteFooter() {
  return (
    <HStack
      as="footer"
      gap={6}
      paddingInline={6}
      paddingBlock={6}
      wrap="wrap"
      vAlign="center"
      className="mt-auto border-t border-border bg-surface"
    >
      <Text type="supporting" color="secondary">
        {DISCLAIMER_SHORT}
      </Text>
      <Text type="supporting" color="secondary">
        {SOURCES_LINE}
        {photosEnabled() ? ` · ${PHOTO_CREDIT}` : ""}
      </Text>
      <Link href={REPO_URL} isExternalLink newTabLabel="(새 창에서 열림)" type="supporting">
        소스 (GitHub)
      </Link>
    </HStack>
  );
}
