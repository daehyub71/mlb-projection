// 공통 머리글·바닥글·메타데이터·OG 이미지 (SPEC F16·F17·D16·D19·D27) — 레인 C Red 테스트
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { cleanup, render, screen, within } from "@testing-library/react";
import { isValidElement, type ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SiteFooter } from "@/app/components/SiteFooter";
import { SiteHeader } from "@/app/components/SiteHeader";
import { metadata } from "@/app/layout";
import { Providers } from "@/app/providers";
import { PHOTO_CREDIT } from "@/src/photo";
import {
  DISCLAIMER_SHORT,
  NAV,
  REPO_URL,
  SITE_DESCRIPTION,
  SITE_NAME,
  SITE_TITLE,
  SOURCES_LINE,
  type NavKey,
} from "@/src/site";

const WEB = join(import.meta.dirname, "..");
const APP = join(WEB, "app");

// next/og의 실제 PNG 변환(sharp)은 jsdom에서 돌지 않는다 — 넘긴 요소·옵션만 받아 둔다
const ogCalls = vi.hoisted(() => [] as Array<{ element: unknown; options: unknown }>);
vi.mock("next/og", () => ({
  ImageResponse: class FakeImageResponse extends Response {
    constructor(element: unknown, options?: unknown) {
      super("fake-png", { headers: { "content-type": "image/png" } });
      ogCalls.push({ element, options });
    }
  },
}));

function renderChrome(active: NavKey | null) {
  return render(
    <Providers>
      <SiteHeader active={active} />
      <SiteFooter />
    </Providers>,
  );
}

// next/link는 테스트에서 trailingSlash 설정을 모르므로 끝 / 는 있어도 없어도 된다
const sameHref = (actual: string | null, expected: string) =>
  (actual ?? "").replace(/\/$/, "") === expected.replace(/\/$/, "");

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("머리글 (SA 공통)", () => {
  it("banner 안에 사이트 표시: 장식 마크 '27'(aria-hidden) + 이름, 첫 화면으로 가는 링크", () => {
    renderChrome("projections");
    const banner = screen.getByRole("banner");
    const home = within(banner).getByRole("link", { name: SITE_NAME });
    expect(sameHref(home.getAttribute("href"), "/")).toBe(true);
    const marks = Array.from(home.querySelectorAll('[aria-hidden="true"]')).filter(
      (el) => (el.textContent ?? "").trim() === "27",
    );
    expect(marks.length, "사이트 마크 '27'은 장식(aria-hidden)").toBeGreaterThan(0);
    expect(home).toHaveTextContent(SITE_NAME);
  });

  it("주요 메뉴: 예측·검증·소개 순서와 주소", () => {
    renderChrome(null);
    const nav = screen.getByRole("navigation", { name: "주요 메뉴" });
    const links = NAV.map((item) => within(nav).getByRole("link", { name: item.label }));
    links.forEach((link, i) => {
      expect(sameHref(link.getAttribute("href"), NAV[i]!.href), link.outerHTML).toBe(true);
      if (i > 0) expect(links[i - 1]!.compareDocumentPosition(link) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    });
  });

  for (const item of NAV) {
    it(`현재 화면(${item.label})만 aria-current="page"`, () => {
      renderChrome(item.key);
      const current = screen.getAllByRole("link", { current: "page" });
      expect(current).toHaveLength(1);
      expect(current[0]).toHaveTextContent(item.label);
    });
  }

  it("현재 화면이 메뉴에 없으면 aria-current가 없다", () => {
    renderChrome(null);
    expect(screen.queryAllByRole("link", { current: "page" })).toHaveLength(0);
  });

  it("머리글은 화면 위에 붙는다 (sticky, SA 공통)", () => {
    // jsdom은 StyleX·Tailwind CSS를 계산하지 않아 소스로 확인한다
    const source = readFileSync(join(APP, "components/SiteHeader.tsx"), "utf-8");
    expect(source).toMatch(/sticky/i);
  });
});

describe("바닥글 (SPEC F16·D16·D27)", () => {
  it("contentinfo 안에 고지·출처+사진 저작권(한 줄)·깃허브 링크", () => {
    renderChrome("validation");
    const footer = screen.getByRole("contentinfo");
    expect(within(footer).getByText(DISCLAIMER_SHORT)).toBeInTheDocument();
    const sources = within(footer).getByText(new RegExp(SOURCES_LINE));
    expect(sources).toHaveTextContent(PHOTO_CREDIT);
    const repo = within(footer).getByRole("link", { name: /GitHub/ });
    expect(repo).toHaveAttribute("href", REPO_URL);
  });

  it("NEXT_PUBLIC_PLAYER_PHOTOS=off면 사진 저작권을 감춘다 (출처는 남는다)", () => {
    vi.stubEnv("NEXT_PUBLIC_PLAYER_PHOTOS", "off");
    renderChrome("about");
    expect(screen.queryAllByText(/선수 사진/)).toHaveLength(0);
    expect(within(screen.getByRole("contentinfo")).getByText(new RegExp(SOURCES_LINE))).toBeInTheDocument();
  });
});

describe("Astryx 규칙 (web/.claude/CLAUDE.md) — 레인 C 소스", () => {
  /** 주석을 뺀 소스 (주석에 DESIGN 색 이름을 적는 것은 괜찮다) */
  const code = (path: string) =>
    readFileSync(path, "utf-8")
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/(^|[^:])\/\/.*$/gm, "$1");

  const laneFiles = (): string[] => {
    const files = [join(APP, "components/SiteHeader.tsx"), join(APP, "components/SiteFooter.tsx"), join(APP, "layout.tsx")];
    for (const dir of ["validation", "about"]) {
      for (const name of readdirSync(join(APP, dir), { recursive: true }) as string[]) {
        if (/\.(tsx|ts)$/.test(name)) files.push(join(APP, dir, name));
      }
    }
    return files;
  };

  it("<div>·<span>·style={…}·hex 색·임의 Tailwind 값·.css 가져오기가 없다", () => {
    for (const file of laneFiles()) {
      const src = code(file);
      const rel = file.slice(WEB.length + 1);
      expect(src, `${rel}: <div>`).not.toMatch(/<div[\s>]/);
      expect(src, `${rel}: <span>`).not.toMatch(/<span[\s>]/);
      expect(src, `${rel}: style={…}`).not.toMatch(/\bstyle=\{/);
      expect(src, `${rel}: hex 색`).not.toMatch(/#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b/);
      expect(src, `${rel}: 임의 Tailwind 값`).not.toMatch(/\b[a-z][a-z0-9-]*-\[[^\]]+\]/);
      if (!rel.endsWith("layout.tsx")) expect(src, `${rel}: .css 가져오기`).not.toMatch(/import\s+["'][^"']+\.css["']/);
    }
  });
});

describe("메타데이터 (app/layout.tsx, SPEC F17)", () => {
  it("제목 틀: 기본은 사이트 제목, 하위 화면은 '%s'에 사이트 이름을 붙인다 · 설명", () => {
    const title = metadata.title as unknown;
    expect(typeof title, "title은 { default, template }").toBe("object");
    const t = title as { default?: unknown; template?: unknown };
    expect(t.default).toBe(SITE_TITLE);
    expect(String(t.template)).toContain("%s");
    expect(String(t.template)).toContain(SITE_NAME);
    expect(metadata.description).toBe(SITE_DESCRIPTION);
  });

  it("Open Graph(웹사이트·한국어·사이트 이름·제목·설명)와 트위터 큰 이미지 카드", () => {
    const og = metadata.openGraph as unknown as Record<string, unknown> | undefined;
    expect(og, "openGraph").toBeDefined();
    expect(og!.type).toBe("website");
    expect(og!.locale).toBe("ko_KR");
    expect(og!.siteName).toBe(SITE_NAME);
    expect(JSON.stringify(og!.title)).toContain(SITE_TITLE);
    expect(og!.description).toBe(SITE_DESCRIPTION);
    const tw = metadata.twitter as unknown as Record<string, unknown> | undefined;
    expect(tw, "twitter").toBeDefined();
    expect(tw!.card).toBe("summary_large_image");
  });

  it("OG 이미지: og-bg.jpg 위에 사이트 제목을 코드로 얹는다 (D19) — 안 되면 og-bg.jpg 그대로", async () => {
    const dynamicOg = ["opengraph-image.tsx", "opengraph-image.ts"].map((f) => join(APP, f)).find((p) => existsSync(p));
    const staticOg = ["opengraph-image.jpg", "opengraph-image.png"].map((f) => join(APP, f)).find((p) => existsSync(p));
    const bg = readFileSync(join(WEB, "public/images/og-bg.jpg"));

    if (dynamicOg === undefined) {
      // 대체 경로: 글자 없는 배경 그대로 (보고 대상)
      if (staticOg === undefined) {
        const og = metadata.openGraph as unknown as { images?: unknown } | undefined;
        const tw = metadata.twitter as unknown as { images?: unknown } | undefined;
        expect(JSON.stringify(og?.images ?? null), "openGraph.images").toContain("/images/og-bg.jpg");
        expect(JSON.stringify(tw?.images ?? null), "twitter.images").toContain("/images/og-bg.jpg");
      }
      return;
    }

    const mod = (await import(/* @vite-ignore */ dynamicOg)) as {
      default: (...args: unknown[]) => unknown;
      size?: unknown;
      contentType?: unknown;
      alt?: unknown;
    };
    expect(mod.size).toEqual({ width: 1200, height: 630 });
    expect(mod.contentType).toBe("image/png");
    expect(typeof mod.alt === "string" && mod.alt.length > 0, "alt").toBe(true);

    ogCalls.length = 0;
    await mod.default();
    expect(ogCalls, "ImageResponse(next/og)를 한 번 만든다").toHaveLength(1);
    const { element, options } = ogCalls[0]!;
    const tree = collect(element);

    // 제목은 코드로 쓴 글자다
    expect(tree.text.replace(/\s+/g, "")).toContain(SITE_TITLE.replace(/\s+/g, ""));
    // 배경은 og-bg.jpg 바이트(데이터 URL이나 버퍼)
    expect(tree.images.some((src) => isSameImage(src, bg)), "og-bg.jpg 배경").toBe(true);
    // 글꼴: Noto Sans KR, woff (satori는 woff2를 읽지 못한다)
    const fonts = ((options as { fonts?: Array<{ name?: unknown; data?: unknown }> } | undefined)?.fonts ?? []).filter(
      (f) => /noto sans kr/i.test(String(f.name)),
    );
    expect(fonts.length, "Noto Sans KR 글꼴").toBeGreaterThan(0);
    for (const f of fonts) {
      const bytes = toBytes(f.data);
      expect(bytes, "글꼴 데이터").not.toBeNull();
      expect(String.fromCharCode(...bytes!.slice(0, 4)), "woff 서명").toBe("wOFF");
    }
  });

  it("OG 이미지 주소가 이미지 확장자로 끝난다 — 정적 export의 확장자 없는 파일은 호스팅이 이미지로 내보내지 못할 수 있다", async () => {
    const dynamicOg = ["opengraph-image.tsx", "opengraph-image.ts"].map((f) => join(APP, f)).find((p) => existsSync(p));
    if (dynamicOg === undefined) return; // 정적 파일(opengraph-image.jpg/png)은 이름에 확장자가 있다
    const mod = (await import(/* @vite-ignore */ dynamicOg)) as {
      generateImageMetadata?: () => Array<{ id: unknown; contentType?: unknown }>;
      contentType?: unknown;
    };
    expect(typeof mod.generateImageMetadata, "generateImageMetadata로 id(파일 이름)를 준다").toBe("function");
    const items = mod.generateImageMetadata!();
    expect(items).toHaveLength(1);
    for (const item of items) {
      expect(String(item.id)).toMatch(/\.png$/);
      expect(item.contentType ?? mod.contentType).toBe("image/png");
    }
  });
});

// ---------------------------------------------------------------------------
// OG 요소 나무 읽기
// ---------------------------------------------------------------------------

function collect(node: unknown, acc: { text: string; images: unknown[] } = { text: "", images: [] }) {
  if (node === null || node === undefined || typeof node === "boolean") return acc;
  if (typeof node === "string" || typeof node === "number") {
    acc.text += `${node} `;
    return acc;
  }
  if (Array.isArray(node)) {
    for (const child of node) collect(child, acc);
    return acc;
  }
  if (isValidElement(node)) {
    const { type, props } = node as ReactElement<Record<string, unknown>>;
    if (typeof type === "function") return collect((type as (p: unknown) => unknown)(props), acc);
    if (type === "img") acc.images.push(props.src);
    const style = props.style as Record<string, unknown> | undefined;
    if (style !== undefined && typeof style.backgroundImage === "string") acc.images.push(style.backgroundImage);
    collect(props.children, acc);
  }
  return acc;
}

function toBytes(x: unknown): Uint8Array | null {
  if (ArrayBuffer.isView(x)) return new Uint8Array(x.buffer, x.byteOffset, x.byteLength);
  if (Object.prototype.toString.call(x) === "[object ArrayBuffer]") return new Uint8Array(x as ArrayBuffer);
  return null;
}

function isSameImage(src: unknown, expected: Buffer): boolean {
  if (typeof src === "string") {
    const match = /base64,([A-Za-z0-9+/=]+)/.exec(src);
    return match !== null && Buffer.from(match[1]!, "base64").equals(expected);
  }
  const bytes = toBytes(src);
  return bytes !== null && Buffer.from(bytes).equals(expected);
}
