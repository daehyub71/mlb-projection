// 시즌 나이 — 그 시즌 6월 30일 기준 만 나이 (파이프라인 mlbproj/age.py와 같은 규칙, SPEC F9)

export function seasonAge(birthDate: string | null | undefined, season: number): number | null {
  if (birthDate == null) return null;
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(birthDate);
  if (match === null) return null;
  const [year, month, day] = [Number(match[1]), Number(match[2]), Number(match[3])];
  const beforeBirthday = month > 6 || (month === 6 && day > 30);
  return season - year - (beforeBirthday ? 1 : 0);
}
