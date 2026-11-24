import next from "eslint-config-next";

const config = [
  ...next,
  { ignores: [".next/**", "out/**", "node_modules/**", "next-env.d.ts", "src/themes/**", "themes/**", "src/astryx-themes.ts"] },
];

export default config;
