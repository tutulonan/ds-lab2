"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Загрузить Titanic и вычислить пропуски и агрегаты по классам."""
    passengers = pd.read_csv(data.csv_path)

    missing = passengers.isna().sum()
    classes = passengers.groupby("Pclass")
    # mean() пропускает неизвестный возраст, не заменяя его нулём.
    mean_ages = classes["Age"].mean()
    survival_rates = classes["Survived"].mean()
    highest_fares = passengers["Fare"].dropna().sort_values(ascending=False).head(5)

    return TitanicSummary(
        row_count=len(passengers),
        missing_by_column={column: int(count) for column, count in missing.items()},
        adults_over_30_count=int((passengers["Age"] > 30).sum()),
        mean_age_by_pclass={int(key): float(value) for key, value in mean_ages.items()},
        survival_rate_by_pclass={
            int(key): float(value) for key, value in survival_rates.items()
        },
        highest_fares=[float(value) for value in highest_fares],
    )
