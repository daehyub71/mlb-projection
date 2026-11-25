// 공유(OG) 이미지 — og-bg.jpg(글자 없는 Codex 배경, D19) 위에 사이트 제목을 코드로 얹는다 (SPEC F17)
// 빌드 때 한 번 PNG로 굳는다(정적 export). Satori는 인라인 style과 px 값만 읽으므로 이 파일만 예외다.
// 글꼴은 사이트와 같은 Noto Sans KR(@fontsource, OFL) — Satori는 woff2를 읽지 못해 woff를 쓴다.
// 한글 묶음(korean)에 숫자·영문이 없을 수 있어 latin 묶음도 같은 이름으로 넣는다.
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";
import { SITE_NAME, SITE_TITLE } from "@/src/site";

export const alt = SITE_TITLE;
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const dynamic = "force-static";

// 정적 export는 이 이미지를 확장자 없는 out/opengraph-image로 쓴다. 정적 호스팅은 확장자로 Content-Type을
// 정하므로 이미지로 내보내지 못할 수 있다 — id에 확장자를 넣어 out/opengraph-image/og.png로 나오게 한다.
// og:image·twitter:image는 Next가 /opengraph-image/og.png?… 로 붙인다.
const OG_IMAGE_ID = "og.png";

export function generateImageMetadata() {
  return [{ id: OG_IMAGE_ID, alt, size, contentType }];
}

const FONT_DIR = "node_modules/@fontsource/noto-sans-kr/files";

async function load(path: string): Promise<Buffer> {
  return readFile(join(process.cwd(), path));
}

export default async function OpengraphImage() {
  const [bg, korean, latin] = await Promise.all([
    load("public/images/og-bg.jpg"),
    load(`${FONT_DIR}/noto-sans-kr-korean-700-normal.woff`),
    load(`${FONT_DIR}/noto-sans-kr-latin-700-normal.woff`),
  ]);
  const bgSrc = `data:image/jpeg;base64,${bg.toString("base64")}`;

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          position: "relative",
          fontFamily: "Noto Sans KR",
          color: "#ffffff",
        }}
      >
        <img src={bgSrc} alt="" width={1200} height={630} style={{ position: "absolute", top: 0, left: 0 }} />
        {/* 글자 대비용 어두운 막 (N10) */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            display: "flex",
            backgroundImage: "linear-gradient(90deg, rgba(10,20,20,0.88) 0%, rgba(10,20,20,0.55) 60%, rgba(10,20,20,0.15) 100%)",
          }}
        />
        <div
          style={{
            position: "relative",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            gap: 28,
            padding: "0 80px",
            width: "100%",
            height: "100%",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: 56,
                height: 56,
                borderRadius: 12,
                background: "#e11d74",
                fontSize: 26,
                fontWeight: 700,
              }}
            >
              27
            </div>
            <div style={{ display: "flex", fontSize: 32, fontWeight: 700, color: "#ff91b1" }}>{SITE_NAME}</div>
          </div>
          <div style={{ display: "flex", fontSize: 72, fontWeight: 700, lineHeight: 1.2, maxWidth: 900 }}>{SITE_TITLE}</div>
        </div>
      </div>
    ),
    {
      ...size,
      fonts: [
        { name: "Noto Sans KR", data: korean, weight: 700, style: "normal" },
        { name: "Noto Sans KR", data: latin, weight: 700, style: "normal" },
      ],
    },
  );
}
