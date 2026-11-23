import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import HomePage from "@/app/page";
import { Providers } from "@/app/providers";

describe("HomePage (M0 자리표시)", () => {
  it("Astryx 테마 안에서 제목·고지·출처를 렌더링한다", () => {
    render(
      <Providers>
        <HomePage />
      </Providers>,
    );
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("2027 MLB");
    expect(screen.getByText(/재미로 보는 예측/)).toBeInTheDocument();
    expect(screen.getByText(/MLB Stats API, Baseball Savant/)).toBeInTheDocument();
  });
});
