import { Heading } from "@astryxdesign/core/Heading";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";
import { SiteFooter } from "../components/SiteFooter";
import { SiteHeader } from "../components/SiteHeader";

// M6 계약 골격 — 레인 C가 소개(SA-5)로 바꾼다
export default function AboutPage() {
  return (
    <VStack gap={0}>
      <SiteHeader active="about" />
      <VStack as="main" gap={3} padding={6}>
        <Heading level={1}>이 예측은 어떻게 만들었나</Heading>
        <Text color="secondary">소개 화면을 준비하고 있습니다.</Text>
      </VStack>
      <SiteFooter />
    </VStack>
  );
}
