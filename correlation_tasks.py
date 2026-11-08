"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Посчитать корреляции с MRI_Count отдельно для мужчин и женщин.

    Пропуски исключаются попарно для каждого коэффициента. Если корреляцию
    определить нельзя (например, признак постоянный), возвращается NaN.
    Если ни одного определённого коэффициента нет, strongest_mri_feature = "".
    """
    observations = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    women = observations.loc[observations["Gender"] == "Female"]
    men = observations.loc[observations["Gender"] == "Male"]

    def mri_correlations(group: pd.DataFrame) -> dict[str, float]:
        matrix = group[features + ["MRI_Count"]].corr(
            method="pearson", min_periods=2
        )
        return {feature: float(matrix.loc[feature, "MRI_Count"]) for feature in features}

    women_correlations = mri_correlations(women)
    men_correlations = mri_correlations(men)
    defined = [
        (feature, value)
        for correlations in (women_correlations, men_correlations)
        for feature, value in correlations.items()
        if pd.notna(value)
    ]
    strongest = max(defined, key=lambda item: abs(item[1]))[0] if defined else ""

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_correlations,
        men_mri_correlation=men_correlations,
        strongest_mri_feature=strongest,
    )
