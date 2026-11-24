// 웹 JSON 불러오기 — 정적 export라 브라우저에서 /data/*.json을 받는다 (파이프라인 `mlbproj export`가 쓴다)
import type { BatTable, PitTable, PlayerDetail, ValidationReport } from "./types";

export const DATA_BASE = "/data";

/** 데이터를 받지 못했거나 모양이 틀렸을 때. status는 HTTP 상태(있으면). */
export class DataError extends Error {
  readonly status: number | undefined;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "DataError";
    this.status = status;
  }
}

export type Fetcher = (url: string) => Promise<Response>;

async function getJson<T>(path: string, fetcher: Fetcher): Promise<T> {
  const url = `${DATA_BASE}/${path}`;
  let res: Response;
  try {
    res = await fetcher(url);
  } catch (err) {
    throw new DataError(`${url}을(를) 받지 못했습니다: ${String(err)}`);
  }
  if (!res.ok) {
    throw new DataError(`${url}: HTTP ${res.status}`, res.status);
  }
  return (await res.json()) as T;
}

function expectRole<T extends { role: string }>(table: T, role: "bat" | "pit", path: string): T {
  if (table.role !== role) {
    throw new DataError(`${path}의 role이 ${role}이 아닙니다 (${table.role})`);
  }
  return table;
}

export async function loadBatters(fetcher: Fetcher = fetch): Promise<BatTable> {
  return expectRole(await getJson<BatTable>("batters.json", fetcher), "bat", "batters.json");
}

export async function loadPitchers(fetcher: Fetcher = fetch): Promise<PitTable> {
  return expectRole(await getJson<PitTable>("pitchers.json", fetcher), "pit", "pitchers.json");
}

export async function loadValidation(fetcher: Fetcher = fetch): Promise<ValidationReport> {
  return getJson<ValidationReport>("validation.json", fetcher);
}

/** 선수 상세 (`/player/?id=` — DESIGN DS1). id가 양의 정수가 아니면 DataError. 없으면 status 404. */
export async function loadPlayer(id: number, fetcher: Fetcher = fetch): Promise<PlayerDetail> {
  if (!Number.isInteger(id) || id <= 0) {
    throw new DataError(`선수 ID가 올바르지 않습니다: ${id}`);
  }
  return getJson<PlayerDetail>(`players/${id}.json`, fetcher);
}

/** `?id=660271` → 660271. 없거나 양의 정수가 아니면 null. */
export function parsePlayerId(search: string | URLSearchParams | null | undefined): number | null {
  if (search == null) return null;
  const params = typeof search === "string" ? new URLSearchParams(search) : search;
  const raw = params.get("id");
  if (raw == null || !/^\d+$/.test(raw)) return null;
  const id = Number(raw);
  return id > 0 ? id : null;
}
