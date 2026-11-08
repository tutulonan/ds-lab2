"""Запуск анализа и создание отчёта: python run_analysis.py."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from correlation_tasks import analyze_brain_correlations
from grader_contracts.correlation_tasks import BrainDataInput
from grader_contracts.pandas_tasks import TitanicInput
from pandas_tasks import analyze_titanic


def plot_brain_data(observations: pd.DataFrame, output_dir: Path) -> None:
    """Сохранить scatter-графики и матрицы корреляций для обеих групп."""
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    sns.set_theme(style="whitegrid", font_scale=0.95)

    # Каждый столбец отвечает одному признаку, каждая строка — одной группе.
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharey=True)
    for row, (gender, color) in enumerate([("Female", "#bc477a"), ("Male", "#397bb0")]):
        group = observations.loc[observations["Gender"] == gender]
        for column, feature in enumerate(features):
            paired = group[[feature, "MRI_Count"]].dropna()
            ax = axes[row, column]
            ax.scatter(paired[feature], paired["MRI_Count"], color=color, alpha=.8,
                       edgecolors="white", linewidths=.5, s=45)
            r = group[[feature, "MRI_Count"]].corr().loc[feature, "MRI_Count"]
            ax.set_title(f"{gender}: r = {r:.3f}, n = {len(paired)}")
            ax.set_xlabel(feature)
            ax.set_ylabel("MRI_Count" if column == 0 else "")
    fig.suptitle("Brain data: features and MRI_Count", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output_dir / "brain_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, gender in zip(axes, ["Female", "Male"]):
        group = observations.loc[observations["Gender"] == gender]
        correlations = group[features + ["MRI_Count"]].corr(method="pearson")
        sns.heatmap(correlations, ax=ax, annot=True, fmt=".2f", cmap="vlag",
                    vmin=-1, vmax=1, center=0, square=True,
                    cbar_kws={"label": "Pearson r", "shrink": .75})
        ax.set_title(f"{gender} (n = {len(group)})")
        ax.tick_params(axis="x", labelrotation=40)
        ax.tick_params(axis="y", labelrotation=0)
    fig.suptitle("Brain data: correlation matrices", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output_dir / "brain_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(titanic, brain, observations: pd.DataFrame, path: Path) -> None:
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    lines = [
        "# Лабораторная работа №2. Pandas и корреляционный анализ",
        "",
        "## 1. Titanic",
        "",
        "Данные загружены из `titanic.csv` функцией `pd.read_csv`. "
        "Пропуски посчитаны через `isna().sum()`, показатели по классам — "
        "через `groupby`. Неизвестный возраст не заменяется нулём.",
        "",
        f"Число пассажиров: **{titanic.row_count}**. "
        f"Число пассажиров строго старше 30 лет: **{titanic.adults_over_30_count}**.",
        "",
        "| Столбец | Количество пропусков |",
        "| --- | ---: |",
    ]
    lines += [f"| {column} | {count} |" for column, count in titanic.missing_by_column.items()]
    lines += [
        "",
        "| Pclass | Средний возраст, лет | Доля выживших |",
        "| --- | ---: | ---: |",
    ]
    for pclass, age in titanic.mean_age_by_pclass.items():
        rate = titanic.survival_rate_by_pclass[pclass]
        lines.append(f"| {pclass} | {age:.3f} | {rate:.6f} ({rate:.2%}) |")
    lines += [
        "",
        "Пять наибольших значений Fare по убыванию: "
        + ", ".join(f"{fare:.4f}" for fare in titanic.highest_fares) + ".",
        "Повторяющиеся тарифы сохранены: нужны пять значений, а не пять уникальных тарифов.",
        "",
        "В этой выборке средний возраст и доля выживших выше в первом классе. "
        "Большинство пропусков приходится на Cabin; пропуски Age исключены только "
        "из расчёта возраста, а не из всей таблицы.",
        "",
        "## 2. Корреляционный анализ",
        "",
        "Файл `brainsize.txt` прочитан с разделителем tab. Обозначения `NA` "
        "распознаны как пропуски. Анализ проведён отдельно для Female и Male.",
        "",
        f"Женщин: **{brain.women_count}**, мужчин: **{brain.men_count}**.",
        "",
        "Использован коэффициент Пирсона: "
        "`r = sum((x-x_mean)*(y-y_mean)) / "
        "sqrt(sum((x-x_mean)**2)*sum((y-y_mean)**2))`.",
        "Он показывает направление и силу линейной связи. "
        "При расчёте каждого коэффициента исключаются только строки, "
        "где отсутствует один из двух участвующих показателей.",
        "",
        "| Признак | r, женщины | Число пар | r, мужчины | Число пар |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for feature in features:
        counts = {}
        for gender in ["Female", "Male"]:
            group = observations.loc[observations["Gender"] == gender]
            counts[gender] = len(group[[feature, "MRI_Count"]].dropna())
        lines.append(
            f"| {feature} | {brain.women_mri_correlation[feature]:.6f} | {counts['Female']} "
            f"| {brain.men_mri_correlation[feature]:.6f} | {counts['Male']} |"
        )
    candidates = [
        (gender, feature, value)
        for gender, correlations in [("женщины", brain.women_mri_correlation),
                                     ("мужчины", brain.men_mri_correlation)]
        for feature, value in correlations.items()
        if pd.notna(value)
    ]
    strongest_group, strongest_feature, strongest_r = max(candidates, key=lambda item: abs(item[2]))
    lines += [
        "",
        f"Наибольший модуль корреляции с MRI_Count: **{strongest_feature}**, "
        f"группа — **{strongest_group}**, `r = {strongest_r:.6f}`. "
        f"Функция возвращает `strongest_mri_feature = \"{brain.strongest_mri_feature}\"`.",
        "",
        "У мужчин самая выраженная положительная линейная связь MRI_Count — с PIQ; "
        "у женщин — с Weight. Связь MRI_Count с Weight у мужчин близка к нулю "
        "и слегка отрицательна. Коэффициенты описывают эту небольшую выборку; "
        "корреляция сама по себе не доказывает причинную связь. "
        "Статистическая значимость различий между группами здесь не проверялась.",
        "",
        "### Диаграммы рассеяния",
        "",
        "![Scatter-графики признаков и MRI_Count](brain_scatter.png)",
        "",
        "Верхняя строка — женщины, нижняя — мужчины. "
        "Подписи содержат коэффициент r и число полных пар n. "
        "По горизонтали — выбранный признак, по вертикали — MRI_Count.",
        "",
        "### Матрицы корреляций",
        "",
        "![Heatmap корреляций по группам](brain_heatmaps.png)",
        "",
        "У обеих матриц одинаковая цветовая шкала от -1 до 1. "
        "Диагональные значения равны 1, поскольку признак сравнивается с самим собой; "
        "сам MRI_Count не участвует в выборе strongest_mri_feature.",
        "",
        "## 3. Запуск и проверка",
        "",
        "Из папки проекта:",
        "",
        "```bash",
        "python -m pip install -r requirements.txt",
        "python -m unittest discover -s tests -v",
        "python run_analysis.py",
        "```",
        "",
        "Тесты находятся в `tests/`. Они проверяют агрегаты на небольших таблицах "
        "и на приложенных данных, границу возраста 30 лет, пропуски, повторяющиеся "
        "тарифы, пустые таблицы, постоянные признаки и выбор максимума по модулю "
        "из обеих групп. Корреляции небольших таблиц сверяются с независимым "
        "расчётом по формуле Пирсона.",
        "",
        "Для неопределённой корреляции возвращается NaN. Если все коэффициенты "
        "неопределены, strongest_mri_feature — пустая строка. "
        "Таблица без строк, но с заголовками поддерживается. Отсутствие файла "
        "или обязательных столбцов вызывает ошибку.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "results"
    output_dir.mkdir(exist_ok=True)
    titanic = analyze_titanic(TitanicInput(str(root / "titanic.csv")))
    brain = analyze_brain_correlations(BrainDataInput(str(root / "brainsize.txt")))
    observations = pd.read_csv(root / "brainsize.txt", sep="\t", na_values=["NA"])
    plot_brain_data(observations, output_dir)
    write_report(titanic, brain, observations, output_dir / "report.md")
    print(f"Titanic: {titanic.row_count} пассажиров, старше 30 лет: {titanic.adults_over_30_count}")
    print(f"Brainsize: мужчин {brain.men_count}, женщин {brain.women_count}")
    print(f"Наиболее связанный с MRI_Count признак: {brain.strongest_mri_feature}")
    print("Отчёт и графики сохранены в results")


if __name__ == "__main__":
    main()
