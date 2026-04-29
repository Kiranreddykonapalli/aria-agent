"""
Generates a realistic synthetic Florida county health dataset.
67 counties x 6 years (2019-2024) = 402 rows.

Design principles:
  - Each county has a stable latent health rank that shifts slightly year-to-year.
  - All metrics are rank-correlated: worse rank -> worse health values.
  - Secular trends: obesity/diabetes worsen gradually; income rises with inflation;
    physician supply slowly erodes; graduation rates inch upward.
  - COVID signal: mental_health_days spike in 2020-2021, unemployment spikes in 2020,
    uninsured_rate rises in 2020 then falls as coverage expanded post-pandemic.
  - health_outcome_rank and health_factor_rank are re-computed within each year
    from the generated metrics so they reflect actual within-year standing.

Run once: python generate_synthetic_data.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

FLORIDA_COUNTIES = [
    "Alachua", "Baker", "Bay", "Bradford", "Brevard", "Broward", "Calhoun",
    "Charlotte", "Citrus", "Clay", "Collier", "Columbia", "DeSoto", "Dixie",
    "Duval", "Escambia", "Flagler", "Franklin", "Gadsden", "Gilchrist",
    "Glades", "Gulf", "Hamilton", "Hardee", "Hendry", "Hernando", "Highlands",
    "Hillsborough", "Holmes", "Indian River", "Jackson", "Jefferson", "Lafayette",
    "Lake", "Lee", "Leon", "Levy", "Liberty", "Madison", "Manatee", "Marion",
    "Martin", "Miami-Dade", "Monroe", "Nassau", "Okaloosa", "Okeechobee",
    "Orange", "Osceola", "Palm Beach", "Pasco", "Pinellas", "Polk", "Putnam",
    "St. Johns", "St. Lucie", "Santa Rosa", "Sarasota", "Seminole", "Sumter",
    "Suwannee", "Taylor", "Union", "Volusia", "Wakulla", "Walton", "Washington",
]

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

N = len(FLORIDA_COUNTIES)  # 67

# 2024 population anchors (real census estimates); earlier years scaled by growth rate.
POP_2024 = {
    "Miami-Dade": 2_701_767, "Broward": 1_952_778, "Palm Beach": 1_492_191,
    "Hillsborough": 1_459_762, "Orange": 1_424_645, "Pinellas": 959_107,
    "Duval": 995_567, "Lee": 760_822, "Polk": 724_777, "Brevard": 606_612,
    "Leon": 292_032, "Alachua": 278_468, "Liberty": 7_542,
}

# Florida avg annual population growth ~1.8% (higher in growth counties, lower in rural)
COUNTY_GROWTH_RATE = rng.uniform(0.005, 0.030, N)


def base_population(year: int) -> np.ndarray:
    """Scale 2024 population back by compound growth for prior years."""
    years_back = 2024 - year
    base = rng.lognormal(mean=11.5, sigma=1.4, size=N).astype(int)
    base = np.clip(base, 7_000, 2_800_000)
    for county, pop in POP_2024.items():
        idx = FLORIDA_COUNTIES.index(county)
        base[idx] = pop
    return np.round(base / (1 + COUNTY_GROWTH_RATE) ** years_back).astype(int)


# --- Stable county latent health rank (1=best, 67=worst) ---
# Represents long-run structural health advantage/disadvantage.
latent_rank = rng.permutation(N) + 1  # integers 1..67

# Year-specific adjustments on top of secular trends.
# Keys: additive deltas applied to generated metrics for that year.
YEAR_ADJ = {
    #            mhd    phd    unemp   unins
    2019: dict(mhd=0.0,  phd=0.0,  unemp=0.000, unins=0.000),
    2020: dict(mhd=1.5,  phd=0.5,  unemp=0.040, unins=0.012),   # COVID-19
    2021: dict(mhd=1.0,  phd=0.3,  unemp=0.012, unins=0.004),   # ongoing COVID
    2022: dict(mhd=0.3,  phd=0.1,  unemp=0.000, unins=-0.008),  # recovery
    2023: dict(mhd=0.1,  phd=0.05, unemp=-0.003, unins=-0.012),
    2024: dict(mhd=0.0,  phd=0.0,  unemp=-0.004, unins=-0.014),
}

# Secular trends: delta per year from 2019 baseline.
def secular(year: int) -> dict:
    t = year - 2019  # 0..5
    return dict(
        obesity   = t * 0.005,            # +0.5pp/yr
        diabetes  = t * 0.003,            # +0.3pp/yr
        mhd_base  = t * 0.05,            # slight secular worsening
        phd_base  = t * 0.03,
        pcp       = t * -0.8,            # physician supply erosion
        income    = (1.025 ** t),        # ~2.5%/yr nominal income growth
        grad      = t * 0.002,           # slow graduation rate improvement
    )


def rank_norm(rank: np.ndarray) -> np.ndarray:
    """Normalise rank to [0, 1]; 0 = best county, 1 = worst county."""
    return (rank - 1) / 66


def rerank(series: pd.Series, ascending: bool = True) -> np.ndarray:
    """Rank a series within a year; ascending=True means low value -> rank 1."""
    return series.rank(method="first", ascending=ascending).astype(int).values


rows = []

for year in YEARS:
    adj = YEAR_ADJ[year]
    sec = secular(year)
    rn = rank_norm(latent_rank)

    # Small year-to-year rank drift (counties shuffle ±4 positions).
    drift = rng.integers(-4, 5, size=N)
    year_rank = np.clip(latent_rank + drift, 1, N)
    rn_y = rank_norm(year_rank)

    pop = base_population(year)

    uninsured_rate = np.clip(
        rng.normal(0.12 + 0.14 * rn_y, 0.025, N) + adj["unins"], 0.04, 0.38
    ).round(3)

    obesity_rate = np.clip(
        rng.normal(0.28 + 0.12 * rn_y + sec["obesity"], 0.03, N), 0.16, 0.50
    ).round(3)

    diabetes_rate = np.clip(
        rng.normal(0.11 + 0.09 * rn_y + sec["diabetes"], 0.02, N), 0.05, 0.27
    ).round(3)

    mental_health_days = np.clip(
        rng.normal(3.5 + 2.0 * rn_y + sec["mhd_base"] + adj["mhd"], 0.5, N), 2.0, 9.0
    ).round(1)

    physical_health_days = np.clip(
        rng.normal(3.0 + 2.5 * rn_y + sec["phd_base"] + adj["phd"], 0.6, N), 1.5, 9.0
    ).round(1)

    primary_care_physicians_rate = np.clip(
        rng.normal(65 - 35 * rn_y + sec["pcp"], 12, N), 8, 130
    ).round(1)

    median_household_income = np.clip(
        rng.normal((62_000 - 22_000 * rn_y) * sec["income"], 7_000, N), 26_000, 115_000
    ).astype(int)

    high_school_graduation_rate = np.clip(
        rng.normal(0.88 - 0.16 * rn_y + sec["grad"], 0.04, N), 0.60, 0.99
    ).round(3)

    unemployment_rate = np.clip(
        rng.normal(0.04 + 0.06 * rn_y + adj["unemp"], 0.012, N), 0.02, 0.20
    ).round(3)

    # Build a composite health outcome score to derive within-year ranks.
    # Lower is healthier (mirrors CHR methodology).
    composite = (
        uninsured_rate * 2.0
        + obesity_rate * 1.5
        + diabetes_rate * 2.0
        + mental_health_days * 0.3
        + physical_health_days * 0.3
        + unemployment_rate * 2.0
        - primary_care_physicians_rate * 0.01
        - median_household_income / 100_000
        - high_school_graduation_rate * 1.0
    )

    df_year = pd.DataFrame({
        "county":                       FLORIDA_COUNTIES,
        "year":                         year,
        "population":                   pop,
        "uninsured_rate":               uninsured_rate,
        "obesity_rate":                 obesity_rate,
        "diabetes_rate":                diabetes_rate,
        "mental_health_days":           mental_health_days,
        "physical_health_days":         physical_health_days,
        "primary_care_physicians_rate": primary_care_physicians_rate,
        "median_household_income":      median_household_income,
        "high_school_graduation_rate":  high_school_graduation_rate,
        "unemployment_rate":            unemployment_rate,
    })

    # Rank within this year (1 = best health outcome).
    df_year["health_outcome_rank"] = rerank(pd.Series(composite), ascending=True)

    factor_composite = (
        uninsured_rate * 1.5
        + obesity_rate * 1.5
        + diabetes_rate * 1.5
        + unemployment_rate * 1.5
        - primary_care_physicians_rate * 0.01
        - median_household_income / 100_000
        - high_school_graduation_rate * 1.0
    )
    df_year["health_factor_rank"] = rerank(pd.Series(factor_composite), ascending=True)

    rows.append(df_year)

df = pd.concat(rows, ignore_index=True)

out = "data/raw/florida_health_2024.csv"
df.to_csv(out, index=False)

print(f"Saved {len(df)} rows -> {out}  ({df['county'].nunique()} counties x {df['year'].nunique()} years)")
print()

# COVID signal check
mhd = df.groupby("year")["mental_health_days"].mean().round(2)
print("Mean mental_health_days by year (COVID spike expected in 2020-2021):")
print(mhd.to_string())
print()

# Secular trend check
ob = df.groupby("year")["obesity_rate"].mean().round(4)
print("Mean obesity_rate by year (should trend upward):")
print(ob.to_string())
print()

print(df.describe().round(3).to_string())
