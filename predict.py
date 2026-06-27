"""
Predicción de partidos internacionales con Dixon-Coles + Elo.

Modelos:
  - Elo con margen de victoria (segunda opinión / fuerza global).
  - Dixon-Coles: ataque/defensa por equipo estimados por máxima verosimilitud,
    con ventaja de campo, corrección de marcadores bajos (rho) y
    decaimiento temporal (los partidos recientes pesan más).

Uso:
  python predict.py                          # partidos de hoy en results.csv
  python predict.py --date 2026-06-20        # partidos de una fecha
  python predict.py --home Germany --away "Ivory Coast" --neutral
  python predict.py --validate               # backtest honesto (train/test temporal)
"""

from __future__ import annotations

import argparse
import math
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# --- Elo ---
INITIAL_ELO = 1500
K_FACTOR = 40
HOME_ELO_BONUS = 65

# --- Dixon-Coles ---
DC_YEARS = 10          # ventana de histórico usada para el ajuste
DC_MIN_MATCHES = 8     # mínimo de partidos para incluir a un equipo
HALF_LIFE_DAYS = 730   # vida media del decaimiento temporal (~2 años)
L2_REG = 0.01          # regularización para estabilizar equipos con pocos datos
MAX_GOALS = 10

# --- Validación ---
TEST_DAYS = 365        # tamaño del set de test en el backtest


def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_matches = pd.read_csv(path, na_values=["NA"])
    all_matches["date"] = pd.to_datetime(all_matches["date"])
    played = all_matches.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    played["neutral"] = played["neutral"].astype(str).str.upper() == "TRUE"
    return all_matches, played.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Elo con margen de victoria
# ---------------------------------------------------------------------------
def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _margin_multiplier(goal_diff: int) -> float:
    diff = abs(goal_diff)
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return (11 + diff) / 8.0


def build_elo(played: pd.DataFrame) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for row in played.itertuples(index=False):
        home_r = ratings.get(row.home_team, INITIAL_ELO)
        away_r = ratings.get(row.away_team, INITIAL_ELO)
        bonus = 0 if row.neutral else HOME_ELO_BONUS

        exp_home = _expected_score(home_r + bonus, away_r)
        if row.home_score > row.away_score:
            actual = 1.0
        elif row.home_score < row.away_score:
            actual = 0.0
        else:
            actual = 0.5

        k = K_FACTOR * _margin_multiplier(row.home_score - row.away_score)
        delta = k * (actual - exp_home)
        ratings[row.home_team] = home_r + delta
        ratings[row.away_team] = away_r - delta
    return ratings


def elo_win_prob(home: str, away: str, neutral: bool, ratings: dict[str, float]) -> float:
    home_r = ratings.get(home, INITIAL_ELO)
    away_r = ratings.get(away, INITIAL_ELO)
    bonus = 0 if neutral else HOME_ELO_BONUS
    return _expected_score(home_r + bonus, away_r)


# ---------------------------------------------------------------------------
# Dixon-Coles
# ---------------------------------------------------------------------------
class DixonColes:
    def __init__(self) -> None:
        self.teams: list[str] = []
        self.index: dict[str, int] = {}
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv: float = 0.0
        self.rho: float = 0.0

    def fit(self, played: pd.DataFrame, ref_date: pd.Timestamp) -> "DixonColes":
        cutoff = ref_date - pd.DateOffset(years=DC_YEARS)
        df = played[(played["date"] >= cutoff) & (played["date"] <= ref_date)].copy()

        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        keep = set(counts[counts >= DC_MIN_MATCHES].index)
        df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)]

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        if n == 0:
            raise ValueError("No hay suficientes partidos para ajustar Dixon-Coles.")

        h_idx = df["home_team"].map(self.index).to_numpy()
        a_idx = df["away_team"].map(self.index).to_numpy()
        hg = df["home_score"].to_numpy()
        ag = df["away_score"].to_numpy()
        neutral = df["neutral"].to_numpy()

        age_days = (ref_date - df["date"]).dt.days.to_numpy()
        xi = math.log(2) / HALF_LIFE_DAYS
        weights = np.exp(-xi * age_days)

        # params: [attack(n), defense(n), home_adv, rho]
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])

        def neg_log_lik(params: np.ndarray) -> float:
            attack = params[:n]
            defense = params[n : 2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]

            log_lh = (~neutral) * home_adv + attack[h_idx] - defense[a_idx]
            log_la = attack[a_idx] - defense[h_idx]
            lam = np.exp(log_lh)
            mu = np.exp(log_la)

            tau = np.ones(len(df))
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            tau = np.clip(tau, 1e-10, None)

            ll = hg * log_lh - lam + ag * log_la - mu + np.log(tau)
            penalty = L2_REG * (np.sum(attack**2) + np.sum(defense**2))
            return -np.sum(weights * ll) + penalty

        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.2, 0.2)]
        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200})

        params = res.x
        attack = params[:n] - params[:n].mean()
        defense = params[n : 2 * n] - params[n : 2 * n].mean()
        self.attack = {t: float(attack[i]) for t, i in self.index.items()}
        self.defense = {t: float(defense[i]) for t, i in self.index.items()}
        self.home_adv = float(params[2 * n])
        self.rho = float(params[2 * n + 1])
        return self

    def lambdas(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        ah = self.attack.get(home, 0.0)
        aa = self.attack.get(away, 0.0)
        dh = self.defense.get(home, 0.0)
        da = self.defense.get(away, 0.0)
        adv = 0.0 if neutral else self.home_adv
        lam = math.exp(adv + ah - da)
        mu = math.exp(aa - dh)
        return lam, mu

    def score_matrix(self, home: str, away: str, neutral: bool) -> np.ndarray:
        lam, mu = self.lambdas(home, away, neutral)
        home_p = poisson.pmf(np.arange(MAX_GOALS + 1), lam)
        away_p = poisson.pmf(np.arange(MAX_GOALS + 1), mu)
        matrix = np.outer(home_p, away_p)
        matrix[0, 0] *= 1 - lam * mu * self.rho
        matrix[0, 1] *= 1 + lam * self.rho
        matrix[1, 0] *= 1 + mu * self.rho
        matrix[1, 1] *= 1 - self.rho
        return matrix / matrix.sum()


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    return home_win, draw, away_win


def top_scorelines(matrix: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
    flat = [
        (f"{i}-{j}", matrix[i, j])
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
    ]
    flat.sort(key=lambda item: item[1], reverse=True)
    return flat[:n]


# ---------------------------------------------------------------------------
# Predicción e impresión
# ---------------------------------------------------------------------------
def predict_match(home, away, neutral, dc, ratings) -> dict:
    matrix = dc.score_matrix(home, away, neutral)
    lam, mu = dc.lambdas(home, away, neutral)
    home_win, draw, away_win = outcome_probs(matrix)
    return {
        "home": home,
        "away": away,
        "neutral": neutral,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "lam": lam,
        "mu": mu,
        "scorelines": top_scorelines(matrix),
        "elo_home_win": elo_win_prob(home, away, neutral, ratings),
        "home_elo": round(ratings.get(home, INITIAL_ELO)),
        "away_elo": round(ratings.get(away, INITIAL_ELO)),
    }


def print_prediction(r: dict) -> None:
    venue = "neutral" if r["neutral"] else "local"
    print(f"\n{r['home']} vs {r['away']} ({venue})")
    print(f"  Elo: {r['home_elo']} vs {r['away_elo']} "
          f"(victoria local Elo: {r['elo_home_win'] * 100:.0f}%)")
    print(f"  1X2 (Dixon-Coles): local {r['home_win'] * 100:.1f}% | "
          f"empate {r['draw'] * 100:.1f}% | visitante {r['away_win'] * 100:.1f}%")
    print(f"  Goles esperados: {r['lam']:.2f} - {r['mu']:.2f}")
    print("  Marcadores más probables:")
    for scoreline, prob in r["scorelines"]:
        print(f"    {scoreline}  ({prob * 100:.1f}%)")


def fixtures_for_date(all_matches: pd.DataFrame, target: date) -> pd.DataFrame:
    day = pd.Timestamp(target)
    pending = all_matches[
        (all_matches["date"] == day)
        & all_matches["home_score"].isna()
        & all_matches["away_score"].isna()
    ]
    return pending if not pending.empty else all_matches[all_matches["date"] == day]


# ---------------------------------------------------------------------------
# Validación (backtest honesto: train pasado, test futuro)
# ---------------------------------------------------------------------------
def validate(played: pd.DataFrame) -> None:
    split = played["date"].max() - pd.Timedelta(days=TEST_DAYS)
    train = played[played["date"] < split]
    test = played[played["date"] >= split]
    print(f"Train: {len(train)} partidos (hasta {split.date()})")
    print(f"Test:  {len(test)} partidos (desde {split.date()})\n")

    dc = DixonColes().fit(train, ref_date=split)

    base = train.copy()
    base_h = (base["home_score"] > base["away_score"]).mean()
    base_d = (base["home_score"] == base["away_score"]).mean()
    base_a = (base["home_score"] < base["away_score"]).mean()
    baseline = np.array([base_h, base_d, base_a])

    eps = 1e-12
    dc_ll = dc_brier = base_ll = base_brier = 0.0
    dc_hits = base_hits = n = 0

    for row in test.itertuples(index=False):
        matrix = dc.score_matrix(row.home_team, row.away_team, row.neutral)
        probs = np.array(outcome_probs(matrix))
        if row.home_score > row.away_score:
            y = 0
        elif row.home_score == row.away_score:
            y = 1
        else:
            y = 2
        onehot = np.zeros(3)
        onehot[y] = 1.0

        dc_ll -= math.log(max(probs[y], eps))
        dc_brier += float(np.sum((probs - onehot) ** 2))
        dc_hits += int(np.argmax(probs) == y)

        base_ll -= math.log(max(baseline[y], eps))
        base_brier += float(np.sum((baseline - onehot) ** 2))
        base_hits += int(np.argmax(baseline) == y)
        n += 1

    print(f"{'Métrica':<18}{'Dixon-Coles':>14}{'Baseline':>14}")
    print(f"{'Log-loss':<18}{dc_ll / n:>14.4f}{base_ll / n:>14.4f}   (menor = mejor)")
    print(f"{'Brier':<18}{dc_brier / n:>14.4f}{base_brier / n:>14.4f}   (menor = mejor)")
    print(f"{'Accuracy':<18}{dc_hits / n:>14.1%}{base_hits / n:>14.1%}   (mayor = mejor)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicción Dixon-Coles + Elo")
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--date", default=str(date.today()))
    parser.add_argument("--home")
    parser.add_argument("--away")
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument("--validate", action="store_true",
                        help="Ejecuta un backtest train/test temporal")
    args = parser.parse_args()

    all_matches, played = load_data(args.csv)

    if args.validate:
        validate(played)
        return

    print("Ajustando modelo Dixon-Coles...")
    ratings = build_elo(played)
    dc = DixonColes().fit(played, ref_date=played["date"].max())

    if args.home and args.away:
        print_prediction(predict_match(args.home, args.away, args.neutral, dc, ratings))
        return

    target = date.fromisoformat(args.date)
    fixtures = fixtures_for_date(all_matches, target)
    if fixtures.empty:
        print(f"No hay partidos en results.csv para {target.isoformat()}.")
        return

    print(f"\nPredicciones para {target.isoformat()} ({len(fixtures)} partido(s))")
    for row in fixtures.itertuples(index=False):
        neutral = str(row.neutral).upper() == "TRUE"
        print_prediction(predict_match(row.home_team, row.away_team, neutral, dc, ratings))


if __name__ == "__main__":
    main()
