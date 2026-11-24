import { Heading } from "@astryxdesign/core/Heading";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import { SiteFooter } from "../components/SiteFooter";
import { SiteHeader } from "../components/SiteHeader";

// M6 계약 골격 — 레인 C가 검증(SA-4, F15)으로 바꾼다
export default function ValidationPage() {
  return (
    <VStack gap={0}>
      <SiteHeader active="validation" />
      <VStack as="main" gap={3} padding={6}>
        <Heading level={1}>예측은 얼마나 맞았나</Heading>
        <Text color="secondary">검증 화면을 준비하고 있습니다.</Text>
      </VStack>
      <SiteFooter />
    </VStack>
  );
}
