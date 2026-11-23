import { Heading } from "@astryxdesign/core/Heading";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/VStack";

// M0 자리표시 — 화면은 DESIGN 시안 합의(M5) 뒤 M6에서 만든다
export default function HomePage() {
  return (
    <VStack gap={3} padding={6}>
      <Heading level={1}>재미로 보는 2027 MLB 선수 성적 예측</Heading>
      <Text color="secondary">준비 중입니다. 재미로 보는 예측이며 실제 성적을 보장하지 않습니다.</Text>
      <Text type="supporting">출처: MLB Stats API, Baseball Savant</Text>
    </VStack>
  );
}
