export interface MatchPrediction {
  match_id: string;
  kickoff_time: string | null;
  home_team: string;
  away_team: string;
  home_short: string;
  away_short: string;
  home_code: number;
  away_code: number;
  home_win: number;
  draw: number;
  away_win: number;
  home_xg: number;
  away_xg: number;
  predicted_at: string;
  finished: boolean;
  retrodiction: boolean;
}

export interface Round {
  season: string;
  gameweek: number;
  matches: MatchPrediction[];
}

const API_URL = process.env.PLPD_API_URL ?? "http://localhost:8000";

export async function getNextRound(): Promise<Round | null> {
  try {
    const response = await fetch(`${API_URL}/predictions/next`, {
      next: { revalidate: 3600 },
      // The free Render instance takes 30-50s to wake from idle.
      signal: AbortSignal.timeout(60_000),
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as Round;
  } catch {
    return null;
  }
}
