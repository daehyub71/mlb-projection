// 검증 (SA-4, SPEC F15) — 서버 쪽 껍데기. 화면 제목(metadata)만 내보내고, 데이터를 읽는 화면은 클라이언트 보기가 그린다.
import type { Metadata } from "next";
import { ValidationView } from "./ValidationView";

export const metadata: Metadata = { title: "검증" };

export default function ValidationPage() {
  return <ValidationView />;
}
