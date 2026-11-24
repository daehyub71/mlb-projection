// 선수 사진 (SPEC D27) — MLB 이미지 서버 주소를 그대로 불러온다. 저장소·배포물에 복사하지 않는다.
// 권리자 요청 시 바로 내릴 수 있게 환경 변수 하나로 끈다: NEXT_PUBLIC_PLAYER_PHOTOS=off

export const PHOTO_CREDIT = "선수 사진 © MLB";

export function photosEnabled(): boolean {
  return process.env.NEXT_PUBLIC_PLAYER_PHOTOS !== "off";
}

/** MLB 공식 헤드샷 주소. 사진이 없는 선수는 서버가 기본 실루엣을 준다. */
export function playerPhotoUrl(playerId: number, width = 213): string {
  return (
    "https://img.mlbstatic.com/mlb-photos/image/upload/" +
    `d_people:generic:headshot:67:current.png/w_${width},q_auto:best/v1/people/${playerId}/headshot/67/current`
  );
}

/** 사진 대신 쓰는 이니셜 배지 글자: "Bobby Witt Jr." → "BW", "Pete Crow-Armstrong" → "PC" */
export function initials(name: string): string {
  const words = name
    .split(/[\s-]+/)
    .filter((w) => w.length > 0 && !/^(jr|sr|ii|iii|iv)\.?$/i.test(w));
  return words
    .slice(0, 2)
    .map((w) => w[0]!.toUpperCase())
    .join("");
}
