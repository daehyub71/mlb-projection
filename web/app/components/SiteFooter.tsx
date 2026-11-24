// 모든 화면 바닥글 — 고지·출처·사진 저작권 (SPEC D16·D27, F16). 문구는 계약, 모양 다듬기는 레인 C
import { HStack } from "@astryxdesign/core/HStack";
import { Link } from "@astryxdesign/core/Link";
import { Text } from "@astryxdesign/core/Text";
import { PHOTO_CREDIT, photosEnabled } from "@/src/photo";
import { DISCLAIMER_SHORT, REPO_URL, SOURCES_LINE } from "@/src/site";

export function SiteFooter() {
  return (
    <HStack as="footer" gap={4} padding={4} wrap="wrap" vAlign="center">
      <Text type="supporting">{DISCLAIMER_SHORT}</Text>
      <Text type="supporting">
        {SOURCES_LINE}
        {photosEnabled() ? ` · ${PHOTO_CREDIT}` : ""}
      </Text>
      <Link href={REPO_URL} isExternalLink type="supporting">
        소스 (GitHub)
      </Link>
    </HStack>
  );
}
