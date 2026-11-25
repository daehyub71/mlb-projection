import { ProjectionsHome } from "./projections/ProjectionsHome";

// 첫 화면 = 2027 예측표 (F13, DESIGN SA-1·SA-3). 데이터는 브라우저에서 받으므로 화면은 클라이언트 컴포넌트다.
export default function HomePage() {
  return <ProjectionsHome />;
}
