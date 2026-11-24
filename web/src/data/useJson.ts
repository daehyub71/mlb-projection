"use client";

// 데이터를 한 번 받아 상태로 들고 있는 훅 — 로딩·오류·성공 세 상태
import { useEffect, useState } from "react";

export type JsonState<T> =
  | { status: "loading" }
  | { status: "ok"; data: T }
  | { status: "error"; message: string; notFound: boolean };

const LOADING = { status: "loading" } as const;

/** load를 key가 바뀔 때마다 한 번 부른다. key가 null이면 부르지 않고 loading으로 둔다. */
export function useJson<T>(load: () => Promise<T>, key: string | null): JsonState<T> {
  // 결과를 받은 key와 함께 둔다 — key가 바뀌면 새 결과가 올 때까지 loading
  const [result, setResult] = useState<{ key: string; state: JsonState<T> } | null>(null);
  useEffect(() => {
    if (key === null) return;
    let alive = true;
    load().then(
      (data) => {
        if (alive) setResult({ key, state: { status: "ok", data } });
      },
      (err: unknown) => {
        if (!alive) return;
        const status = (err as { status?: number }).status;
        const message = err instanceof Error ? err.message : String(err);
        setResult({ key, state: { status: "error", message, notFound: status === 404 } });
      },
    );
    return () => {
      alive = false;
    };
    // load는 key로 대신한다 — 호출부가 매번 새 함수를 넘겨도 다시 받지 않게
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return result !== null && result.key === key ? result.state : LOADING;
}
