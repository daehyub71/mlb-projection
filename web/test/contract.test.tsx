// M6 계약 테스트 — 스키마와 타입 키, 데이터 로더, 숫자 표기, 사진, 공통 머리글·바닥글
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SiteFooter } from "@/app/components/SiteFooter";
import { SiteHeader } from "@/app/components/SiteHeader";
import { Providers } from "@/app/providers";
import { DATA_BASE, DataError, loadBatters, loadPitchers, loadPlayer, loadValidation, parsePlayerId } from "@/src/data/load";
import { NESTED_SCHEMA_KEYS, SCHEMA_KEYS } from "@/src/data/types";
import { fmtBand, fmtCount, fmtDiff, fmtEra, fmtIp, fmtMae, fmtMetric, fmtPct, fmtRate } from "@/src/format";
import { initials, playerPhotoUrl } from "@/src/photo";
import { careerGroup, playerHref } from "@/src/site";

const ROOT = join(import.meta.dirname, "..", "..");
const schema = JSON.parse(readFileSync(join(ROOT, "contracts/web-data.schema.json"), "utf-8")) as {
  $defs: Record<string, { properties: Record<string, { properties?: Record<string, unknown>; oneOf?: unknown }> }>;
};
const sample = (name: string): unknown =>
  JSON.parse(readFileSync(join(ROOT, "tests/fixtures/synthetic/web", name), "utf-8"));

describe("타입 키 = 스키마 속성", () => {
  for (const [def, keys] of Object.entries(SCHEMA_KEYS)) {
    it(def, () => {
      expect(Object.keys(keys).sort()).toEqual(Object.keys(schema.$defs[def]!.properties).sort());
    });
  }
  for (const [path, keys] of Object.entries(NESTED_SCHEMA_KEYS)) {
    it(path, () => {
      const [def, prop] = path.split(".") as [string, string];
      const node = schema.$defs[def]!.properties[prop]!;
      expect(Object.keys(keys).sort()).toEqual(Object.keys(node.properties ?? {}).sort());
    });
  }
});

/** 경로 → 합성 샘플 응답을 주는 가짜 fetch. 모르는 경로는 404 */
function fakeFetch(files: Record<string, unknown>) {
  const calls: string[] = [];
  const fetcher = async (url: string): Promise<Response> => {
    calls.push(url);
    const body = files[url];
    return body === undefined
      ? new Response("not found", { status: 404 })
      : new Response(JSON.stringify(body), { status: 200 });
  };
  return { fetcher, calls };
}

describe("데이터 로더", () => {
  const files = {
    [`${DATA_BASE}/batters.json`]: sample("batters.json"),
    [`${DATA_BASE}/pitchers.json`]: sample("pitchers.json"),
    [`${DATA_BASE}/validation.json`]: sample("validation.json"),
    [`${DATA_BASE}/players/1.json`]: sample("player_bat.json"),
  };

  it("표·검증·선수 상세를 /data에서 받는다", async () => {
    const { fetcher, calls } = fakeFetch(files);
    expect((await loadBatters(fetcher)).rows[0]!.player.name).toBe("Sample Batter");
    expect((await loadPitchers(fetcher)).role).toBe("pit");
    expect((await loadValidation(fetcher)).final_test).toBeNull();
    expect((await loadPlayer(1, fetcher)).bat?.group_error.band80).toEqual({ ops: 0.097 });
    expect(calls).toEqual([
      "/data/batters.json",
      "/data/pitchers.json",
      "/data/validation.json",
      "/data/players/1.json",
    ]);
  });

  it("없는 선수는 status 404인 DataError", async () => {
    const { fetcher } = fakeFetch(files);
    await expect(loadPlayer(999, fetcher)).rejects.toMatchObject({ name: "DataError", status: 404 });
  });

  it("역할이 다른 표는 거부한다", async () => {
    const { fetcher } = fakeFetch({ "/data/batters.json": sample("pitchers.json") });
    await expect(loadBatters(fetcher)).rejects.toBeInstanceOf(DataError);
  });

  it("선수 ID 검사", async () => {
    await expect(loadPlayer(0)).rejects.toBeInstanceOf(DataError);
    expect(parsePlayerId("?id=660271")).toBe(660271);
    expect(parsePlayerId(new URLSearchParams("id=12"))).toBe(12);
    for (const bad of ["", "?id=", "?id=abc", "?id=-3", "?id=1.5", "?id=0", null]) {
      expect(parsePlayerId(bad)).toBeNull();
    }
    expect(playerHref(660271)).toBe("/player/?id=660271");
  });
});

describe("숫자 표기", () => {
  it("비율·ERA·백분율·개수·이닝", () => {
    expect(fmtRate(0.939)).toBe(".939");
    expect(fmtRate(1.0331)).toBe("1.033");
    expect(fmtRate(null)).toBe("—");
    expect(fmtEra(2.8)).toBe("2.80");
    expect(fmtPct(19.75)).toBe("19.8");
    expect(fmtPct(-1.25)).toBe("−1.3");
    expect(fmtCount(503.6)).toBe("504");
    expect(fmtCount(3341)).toBe("3,341");
    expect(fmtIp(76.7)).toBe("76.2"); // 230아웃 = 76과 2/3이닝
    expect(fmtIp(172.3333)).toBe("172.1");
    expect(fmtIp(0)).toBe("0.0");
  });

  it("지표별 표기·범위·차이·평균 오차", () => {
    expect(fmtMetric("ops", 0.898)).toBe(".898");
    expect(fmtMetric("era", 2.82)).toBe("2.82");
    expect(fmtMetric("k_bb_pct", 19.8)).toBe("19.8");
    expect(fmtMetric("pa", 499.8)).toBe("500");
    expect(fmtMetric("ip", 76.7)).toBe("76.2");
    expect(fmtBand("ops", 0.096)).toBe("±.096");
    expect(fmtBand("era", 1.38)).toBe("±1.38");
    expect(fmtBand("k_bb_pct", 5.5)).toBe("±5.5");
    expect(fmtDiff("ops", 0.179)).toBe("+.179");
    expect(fmtDiff("ops", -0.019)).toBe("−.019");
    expect(fmtMae("ops", 0.0615)).toBe(".0615");
    expect(fmtMae("era", 0.8432)).toBe(".843");
    expect(fmtMae("k_bb_pct", 3.6156)).toBe("3.62");
    expect(fmtMae("pa", 72.3687)).toBe("72.4");
    expect(fmtMae("hr", 2.6104)).toBe("2.61");
  });
});

describe("사진·그룹", () => {
  it("MLB 헤드샷 주소와 이니셜", () => {
    expect(playerPhotoUrl(660271)).toContain("/people/660271/headshot/");
    expect(playerPhotoUrl(660271)).toMatch(/^https:\/\/img\.mlbstatic\.com\//);
    expect(initials("Shohei Ohtani")).toBe("SO");
    expect(initials("Bobby Witt Jr.")).toBe("BW");
    expect(initials("Pete Crow-Armstrong")).toBe("PC");
    expect(careerGroup(2)).toBe("seasons_1_2");
    expect(careerGroup(3)).toBe("seasons_3_5");
  });
});

describe("공통 머리글·바닥글", () => {
  it("메뉴 세 개와 현재 화면 표시, 고지·출처·사진 저작권", () => {
    render(
      <Providers>
        <SiteHeader active="validation" />
        <SiteFooter />
      </Providers>,
    );
    const nav = screen.getByRole("navigation", { name: "주요 메뉴" });
    expect(nav).toHaveTextContent("예측");
    expect(screen.getByRole("link", { current: "page" })).toHaveTextContent("검증");
    expect(screen.getByText(/재미로 보는 예측/)).toBeInTheDocument();
    expect(screen.getByText(/MLB Stats API, Baseball Savant/)).toHaveTextContent("선수 사진 © MLB");
  });
});

describe("useJson", () => {
  it("로딩 → 성공, 실패는 notFound와 함께", async () => {
    const { renderHook, waitFor } = await import("@testing-library/react");
    const { useJson } = await import("@/src/data/useJson");
    const ok = renderHook(() => useJson(async () => 42, "a"));
    expect(ok.result.current.status).toBe("loading");
    await waitFor(() => expect(ok.result.current).toEqual({ status: "ok", data: 42 }));
    const bad = renderHook(() =>
      useJson(async () => {
        throw new DataError("없음", 404);
      }, "b"),
    );
    await waitFor(() => expect(bad.result.current).toMatchObject({ status: "error", notFound: true }));
    const idle = renderHook(() => useJson(async () => 1, null));
    expect(idle.result.current.status).toBe("loading");
  });
});

describe("시즌 나이 (mlbproj/age.py와 같은 경우)", () => {
  it("6월 30일 기준", async () => {
    const { seasonAge } = await import("@/src/age");
    expect(seasonAge("1994-07-05", 2027)).toBe(32);
    expect(seasonAge("1994-06-30", 2027)).toBe(33);
    expect(seasonAge("1994-01-01", 2027)).toBe(33);
    expect(seasonAge("1994-12-31", 2027)).toBe(32);
    expect(seasonAge("2000-02-29", 2027)).toBe(27);
    expect(seasonAge(null, 2027)).toBeNull();
    expect(seasonAge("07/05/1994", 2027)).toBeNull();
  });
});
