import type { NextConfig } from "next";

// 정적 export (SPEC D2) — 서버 없이 Vercel 정적 호스팅
const config: NextConfig = {
  output: "export",
  reactStrictMode: true,
  poweredByHeader: false,
  images: { unoptimized: true },
  trailingSlash: true,
};

export default config;
