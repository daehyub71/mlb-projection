import { Heading } from "@astryxdesign/core/Heading";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import { SiteFooter } from "./components/SiteFooter";
import { SiteHeader } from "./components/SiteHeader";
import { SITE_TITLE } from "@/src/site";

// M6 계약 골격 — 레인 A가 table-page 템플릿에서 예측표(SA-1·SA-3, F13)로 바꾼다
export default function HomePage() {
  return (
    <VStack gap={0}>
      <SiteHeader active="projections" />
      <VStack as="main" gap={3} padding={6}>
        <Heading level={1}>{SITE_TITLE}</Heading>
        <Text color="secondary">예측표를 준비하고 있습니다.</Text>
      </VStack>
      <SiteFooter />
    </VStack>
  );
}
