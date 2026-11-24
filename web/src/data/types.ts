// 웹 JSON 타입 — contracts/web-data.schema.json과 짝 (메인 세션만 고친다)
// 스키마를 바꾸면 아래 *_KEYS도 같이 고친다. test/contract.test.ts가 스키마와 키를 대조한다.

export type Role = "bat" | "pit";
export type GroupName = "all" | "seasons_1_2" | "seasons_3_5" | "prev_played" | "prev_absent";
export type CareerGroup = "seasons_1_2" | "seasons_3_5";
export type ModelName = "league" | "baseline" | "weighted" | "extended";
export type BatMetric = "ops" | "pa" | "hr";
export type PitMetric = "era" | "ip" | "k_bb_pct";
export type Metric = BatMetric | PitMetric;
export type BandMetric = "ops" | "era" | "k_bb_pct";
export type Position = "C" | "1B" | "2B" | "3B" | "SS" | "LF" | "CF" | "RF";

export interface Source {
  name: string;
  url: string;
}

export interface Meta {
  schema_version: "1";
  generated_at: string;
  raw_collected_at: string;
  params_version: string;
  params_sha256: string;
  target_season: number;
  input_seasons: number[];
  sources: Source[];
  disclaimer: string;
}

export interface PlayerSummary {
  player_id: number;
  name: string;
  primary_position: string | null; // DH·SS·P·TWP 등
  age: number | null; // 목표 시즌 나이
  team_name: string | null; // 마지막 입력 시즌 팀
  active: boolean; // 표시용: false면 "현역 아님" (SPEC D23)
  last_mlb_season: number;
}

export interface InputWindow {
  input_first: number;
  input_last: number;
  n_seasons: number;
}

export interface BatProjection {
  pa: number;
  hr: number;
  obp: number;
  slg: number;
  ops: number;
}

export interface PitProjection {
  ip: number; // 소수 이닝 (아웃 / 3)
  era: number;
  k_bb_pct: number;
}

export interface BatPrev {
  season: number;
  pa: number;
  hr: number;
  ops: number | null;
}

export interface PitPrev {
  season: number;
  ip: number;
  era: number | null;
  k_bb_pct: number | null;
}

export interface BatRow {
  player: PlayerSummary;
  window: InputWindow;
  sample_pa: number;
  projection: BatProjection;
  prev: BatPrev | null; // 직전 시즌(2026)에 뛰지 않았으면 null
}

export interface PitRow {
  player: PlayerSummary;
  window: InputWindow;
  sample_bf: number;
  sample_ip: number;
  projection: PitProjection;
  prev: PitPrev | null;
}

/** 그룹 → 지표 → 과거 오차 80% 범위 (SPEC D29). 비율 지표만, 값이 없으면 null */
export type Bands = Record<CareerGroup, Partial<Record<BandMetric, number | null>>>;

export interface BatTable {
  meta: Meta;
  role: "bat";
  model: ModelName;
  league_baseline: { ops?: number };
  bands: Bands;
  rows: BatRow[];
}

export interface PitTable {
  meta: Meta;
  role: "pit";
  model: ModelName;
  league_baseline: { era?: number; k_bb_pct?: number };
  bands: Bands;
  rows: PitRow[];
}

export interface BatSeason {
  season: number;
  age: number | null;
  team_name: string | null;
  is_short_season: boolean; // 2020 (60경기)
  pa: number;
  hr: number;
  obp: number | null;
  slg: number | null;
  ops: number | null;
  k_pct: number | null;
  bb_pct: number | null;
  woba: number | null;
  xwoba: number | null;
  barrel_pct: number | null;
  hard_hit_pct: number | null;
  ev_avg: number | null;
}

export interface PitSeason {
  season: number;
  age: number | null;
  team_name: string | null;
  is_short_season: boolean;
  g: number;
  gs: number;
  ip: number;
  era: number | null;
  fip: number | null;
  k_pct: number | null;
  bb_pct: number | null;
  k_bb_pct: number | null;
  xera: number | null;
  xwoba_against: number | null;
  whiff_pct: number | null;
}

export interface GroupError {
  group: GroupName;
  mae: Partial<Record<Metric, number | null>>;
  band80: Partial<Record<BandMetric, number | null>>;
}

export interface BatDetail {
  window: InputWindow;
  sample_pa: number;
  projection: BatProjection;
  prev: BatPrev | null;
  league_baseline: { ops?: number };
  group_error: GroupError;
  history: BatSeason[];
}

export interface PitDetail {
  window: InputWindow;
  sample_bf: number;
  sample_ip: number;
  projection: PitProjection;
  prev: PitPrev | null;
  league_baseline: { era?: number; k_bb_pct?: number };
  group_error: GroupError;
  history: PitSeason[];
}

export interface FieldingSeason {
  season: number;
  oaa: number | null;
  frv_total: number | null;
  frv_range: number | null;
  frv_arm: number | null;
  frv_dp: number | null;
  frv_framing: number | null;
  frv_blocking: number | null;
  frv_throwing: number | null;
  outs_by_pos: Partial<Record<Position, number>>;
}

export interface PlayerInfo {
  player_id: number;
  name: string;
  birth_date: string | null;
  primary_position: string | null;
  bats: string | null;
  throws: string | null;
  active: boolean;
  last_mlb_season: number;
}

export interface PlayerDetail {
  meta: Meta;
  player: PlayerInfo;
  bat: BatDetail | null; // 그 역할 예측 대상이 아니면 null (F12)
  pit: PitDetail | null;
  fielding: FieldingSeason[];
}

export interface EvalRow {
  target_season: number;
  role: Role;
  model: ModelName;
  metric: Metric;
  group: GroupName;
  n: number;
  mae: number | null;
  mae_weighted: number | null;
  baseline_mae_league: number | null;
  baseline_mae_prev: number | null;
  excluded: number;
}

export interface Rerun {
  ran_at: string;
  reason: string;
}

export interface FirstRun {
  ran_at: string;
  mae: Partial<Record<Metric, number | null>>; // 첫 실행의 최종 모델 "all" 그룹 MAE (SPEC D31)
}

export interface FinalTest {
  target_season: number;
  ran_at: string; // 첫 실행 시각
  params_sha256: string;
  reruns: Rerun[];
  first_run: FirstRun | null; // 다시 돌렸을 때만
}

export interface BandRow {
  role: Role;
  metric: BandMetric;
  group: CareerGroup;
  n: number;
  q80: number | null;
  targets: number[];
}

export interface ValidationReport {
  meta: Meta;
  final_test: FinalTest | null;
  rows: EvalRow[];
  bands: BandRow[];
}

// ---------------------------------------------------------------------------
// 스키마 대조용 키 목록 — `satisfies`로 인터페이스 키와 빠짐없이 같게 묶는다
// ---------------------------------------------------------------------------

type Keys<T> = Record<keyof T, true>;

export const SCHEMA_KEYS = {
  Meta: {
    schema_version: true,
    generated_at: true,
    raw_collected_at: true,
    params_version: true,
    params_sha256: true,
    target_season: true,
    input_seasons: true,
    sources: true,
    disclaimer: true,
  } satisfies Keys<Meta>,
  PlayerSummary: {
    player_id: true,
    name: true,
    primary_position: true,
    age: true,
    team_name: true,
    active: true,
    last_mlb_season: true,
  } satisfies Keys<PlayerSummary>,
  InputWindow: { input_first: true, input_last: true, n_seasons: true } satisfies Keys<InputWindow>,
  BatProjection: { pa: true, hr: true, obp: true, slg: true, ops: true } satisfies Keys<BatProjection>,
  PitProjection: { ip: true, era: true, k_bb_pct: true } satisfies Keys<PitProjection>,
  BatPrev: { season: true, pa: true, hr: true, ops: true } satisfies Keys<BatPrev>,
  PitPrev: { season: true, ip: true, era: true, k_bb_pct: true } satisfies Keys<PitPrev>,
  BatRow: { player: true, window: true, sample_pa: true, projection: true, prev: true } satisfies Keys<BatRow>,
  PitRow: {
    player: true,
    window: true,
    sample_bf: true,
    sample_ip: true,
    projection: true,
    prev: true,
  } satisfies Keys<PitRow>,
  BatTable: {
    meta: true,
    role: true,
    model: true,
    league_baseline: true,
    bands: true,
    rows: true,
  } satisfies Keys<BatTable>,
  PitTable: {
    meta: true,
    role: true,
    model: true,
    league_baseline: true,
    bands: true,
    rows: true,
  } satisfies Keys<PitTable>,
  BatSeason: {
    season: true,
    age: true,
    team_name: true,
    is_short_season: true,
    pa: true,
    hr: true,
    obp: true,
    slg: true,
    ops: true,
    k_pct: true,
    bb_pct: true,
    woba: true,
    xwoba: true,
    barrel_pct: true,
    hard_hit_pct: true,
    ev_avg: true,
  } satisfies Keys<BatSeason>,
  PitSeason: {
    season: true,
    age: true,
    team_name: true,
    is_short_season: true,
    g: true,
    gs: true,
    ip: true,
    era: true,
    fip: true,
    k_pct: true,
    bb_pct: true,
    k_bb_pct: true,
    xera: true,
    xwoba_against: true,
    whiff_pct: true,
  } satisfies Keys<PitSeason>,
  GroupError: { group: true, mae: true, band80: true } satisfies Keys<GroupError>,
  BatDetail: {
    window: true,
    sample_pa: true,
    projection: true,
    prev: true,
    league_baseline: true,
    group_error: true,
    history: true,
  } satisfies Keys<BatDetail>,
  PitDetail: {
    window: true,
    sample_bf: true,
    sample_ip: true,
    projection: true,
    prev: true,
    league_baseline: true,
    group_error: true,
    history: true,
  } satisfies Keys<PitDetail>,
  FieldingSeason: {
    season: true,
    oaa: true,
    frv_total: true,
    frv_range: true,
    frv_arm: true,
    frv_dp: true,
    frv_framing: true,
    frv_blocking: true,
    frv_throwing: true,
    outs_by_pos: true,
  } satisfies Keys<FieldingSeason>,
  PlayerDetail: { meta: true, player: true, bat: true, pit: true, fielding: true } satisfies Keys<PlayerDetail>,
  EvalRow: {
    target_season: true,
    role: true,
    model: true,
    metric: true,
    group: true,
    n: true,
    mae: true,
    mae_weighted: true,
    baseline_mae_league: true,
    baseline_mae_prev: true,
    excluded: true,
  } satisfies Keys<EvalRow>,
  BandRow: { role: true, metric: true, group: true, n: true, q80: true, targets: true } satisfies Keys<BandRow>,
  ValidationReport: { meta: true, final_test: true, rows: true, bands: true } satisfies Keys<ValidationReport>,
} as const;

/** 스키마 속성 객체 안쪽의 키 목록 (def 이름 → 경로 → 키) */
export const NESTED_SCHEMA_KEYS = {
  "PlayerDetail.player": {
    player_id: true,
    name: true,
    birth_date: true,
    primary_position: true,
    bats: true,
    throws: true,
    active: true,
    last_mlb_season: true,
  } satisfies Keys<PlayerInfo>,
  "ValidationReport.final_test": {
    target_season: true,
    ran_at: true,
    params_sha256: true,
    reruns: true,
    first_run: true,
  } satisfies Keys<FinalTest>,
} as const;
