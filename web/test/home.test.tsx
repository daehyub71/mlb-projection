import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import HomePage from "@/app/page";
import { Providers } from "@/app/providers";

// M6 계약 골격의 첫 화면 — 레인 A가 예측표로 바꾸며 이 테스트도 바꾼다
describe("HomePage (계약 골격)", () => {
  it("Astryx 테마 안에서 제목·메뉴·고지·출처를 렌더링한다", () => {
    render(
      <Providers>
        <HomePage />
      </Providers>,
    );
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("2027 MLB");
    expect(screen.getByRole("link", { current: "page" })).toHaveTextContent("예측");
    expect(screen.getByText(/재미로 보는 예측/)).toBeInTheDocument();
    expect(screen.getByText(/MLB Stats API, Baseball Savant/)).toBeInTheDocument();
  });
});
