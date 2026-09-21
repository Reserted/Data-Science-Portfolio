"""Generate the synthetic subscriber dataset used by projects/customer-churn.ipynb.

Run `python data/make_sample_data.py` to regenerate `data/subscribers.csv`.
Replace this whole file once you plug in a real dataset.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 3000


def main() -> None:
    plan = RNG.choice(["Basic", "Plus", "Pro"], N, p=[0.5, 0.35, 0.15])
    tenure = RNG.integers(1, 60, N)
    monthly = np.where(plan == "Basic", 9.0, np.where(plan == "Plus", 19.0, 39.0))
    monthly = np.round(monthly + RNG.normal(0, 2, N), 2)
    support = RNG.poisson(1.2, N)
    logins = np.clip(RNG.normal(14, 6, N) - support * 1.5, 0, None).round(0)
    contract = RNG.choice(["Monthly", "Annual"], N, p=[0.7, 0.3])

    # Churn odds rise with support load, fall with tenure, logins, annual contracts.
    logit = (
        -0.5
        + 0.45 * support
        - 0.035 * tenure
        - 0.06 * logins
        - 1.1 * (contract == "Annual")
        + 0.012 * monthly
    )
    churned = RNG.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(N)],
            "plan": plan,
            "contract": contract,
            "tenure_months": tenure,
            "monthly_charge": monthly,
            "support_tickets": support,
            "logins_last_30d": logins.astype(int),
            "churned": churned,
        }
    )
    df.to_csv("data/subscribers.csv", index=False)
    print(f"wrote data/subscribers.csv  rows={len(df)}  churn={df.churned.mean():.1%}")


if __name__ == "__main__":
    main()
