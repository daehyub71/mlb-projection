import { Heading } from "@astryxdesign/core/Heading";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import { SiteFooter } from "../components/SiteFooter";
import { SiteHeader } from "../components/SiteHeader";

// M6 계약 골격 — 레인 B가 선수 상세(SA-2, F14)로 바꾼다. 주소 /player/?id=660271 (DS1)
export default function PlayerPage() {
  return (
    <VStack gap={0}>
      <SiteHeader active="projections" />
      <VStack as="main" gap={3} padding={6}>
        <Heading level={1}>선수 상세</Heading>
        <Text color="secondary">선수 상세를 준비하고 있습니다.</Text>
      </VStack>
      <SiteFooter />
    </VStack>
  );
}
