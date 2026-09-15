// Kickoff timestamps are UTC and the dates shown are UK dates.
const matchDate = new Intl.DateTimeFormat("en-GB", {
  weekday: "short",
  day: "numeric",
  month: "short",
  timeZone: "Europe/London",
});

const generatedAt = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  timeZone: "Europe/London",
});

export function toPercentages(
  home: number,
  draw: number,
  away: number,
): [number, number, number] {
  // Rounding each of the three independently can total 99 or 101.
  const scaled = [home, draw, away].map((value) => value * 100);
  const whole = scaled.map(Math.floor);
  const short = 100 - whole.reduce((total, value) => total + value, 0);
  const byFraction = scaled
    .map((value, index) => ({ index, fraction: value - Math.floor(value) }))
    .sort((a, b) => b.fraction - a.fraction);

  for (let i = 0; i < short; i += 1) {
    whole[byFraction[i].index] += 1;
  }

  return [whole[0], whole[1], whole[2]];
}

export function formatXg(value: number): string {
  return value.toFixed(1);
}

export function formatMatchDate(iso: string | null): string {
  if (iso === null) {
    return "TBC";
  }
  return matchDate.format(new Date(iso));
}

export function formatGeneratedAt(iso: string): string {
  return generatedAt.format(new Date(iso));
}
