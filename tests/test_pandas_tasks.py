"""Примеры и граничные случаи анализа Titanic."""
import csv
import math
from pathlib import Path
import tempfile
import unittest

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary
from pandas_tasks import analyze_titanic


class TitanicTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "titanic.csv"

    def analyze(self, rows, columns=None):
        columns = columns or ["Pclass", "Age", "Survived", "Fare", "Cabin"]
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(rows)
        return analyze_titanic(TitanicInput(str(self.path)))

    def test_counts_and_group_means(self):
        result = self.analyze([
            [1, 20, 1, 10, "A"], [1, 40, 0, 20, ""],
            [2, 30, 1, 30, "B"], [2, "", 1, 40, ""],
            [3, 60, 0, 50, "C"], [3, 10, 0, 60, ""],
        ])
        self.assertIsInstance(result, TitanicSummary)
        self.assertEqual(result.row_count, 6)
        self.assertEqual(result.missing_by_column,
                         {"Pclass": 0, "Age": 1, "Survived": 0, "Fare": 0, "Cabin": 3})
        self.assertEqual(result.adults_over_30_count, 2)
        self.assertEqual(result.mean_age_by_pclass, {1: 30, 2: 30, 3: 35})
        self.assertEqual(result.survival_rate_by_pclass, {1: .5, 2: 1, 3: 0})
        self.assertEqual(result.highest_fares, [60, 50, 40, 30, 20])
        self.assertTrue(all(type(key) is int for key in result.mean_age_by_pclass))
        self.assertTrue(all(type(key) is int for key in result.survival_rate_by_pclass))

    def test_age_boundary_is_strict(self):
        for age, expected in [(0, 0), (29.9, 0), (30, 0), (30.1, 1), (80, 1), ("", 0)]:
            with self.subTest(age=age):
                result = self.analyze([[1, age, 1, 10, ""]])
                self.assertEqual(result.adults_over_30_count, expected)

    def test_missing_age_is_not_zero(self):
        result = self.analyze([[1, 40, 1, 10, ""], [1, "", 0, 20, ""]])
        self.assertEqual(result.mean_age_by_pclass, {1: 40})
        self.assertEqual(result.survival_rate_by_pclass, {1: .5})

    def test_highest_fares_preserve_duplicates(self):
        result = self.analyze([[1, 20, 1, fare, ""] for fare in [5, 100, 20, 100, 0, 80, 40]])
        self.assertEqual(result.highest_fares, [100, 100, 80, 40, 20])

    def test_fewer_than_five_fares_and_missing_values(self):
        for fares, expected in [([2], [2]), ([2, "", 10], [10, 2]), (["", ""], [])]:
            with self.subTest(fares=fares):
                result = self.analyze([[1, 20, 1, fare, ""] for fare in fares])
                self.assertEqual(result.highest_fares, expected)

    def test_all_ages_missing_in_a_class(self):
        result = self.analyze([[2, "", 0, 0, ""], [2, "", 1, 5, ""]])
        self.assertTrue(math.isnan(result.mean_age_by_pclass[2]))
        self.assertEqual(result.adults_over_30_count, 0)
        self.assertEqual(result.survival_rate_by_pclass, {2: .5})

    def test_empty_table(self):
        result = self.analyze([])
        self.assertEqual(result.row_count, 0)
        self.assertEqual(result.adults_over_30_count, 0)
        self.assertEqual(result.mean_age_by_pclass, {})
        self.assertEqual(result.survival_rate_by_pclass, {})
        self.assertEqual(result.highest_fares, [])
        self.assertTrue(all(count == 0 for count in result.missing_by_column.values()))

    def test_attached_dataset(self):
        path = Path(__file__).resolve().parents[1] / "titanic.csv"
        original = path.read_bytes()
        result = analyze_titanic(TitanicInput(str(path)))
        self.assertEqual(result.row_count, 891)
        self.assertEqual(result.adults_over_30_count, 305)
        self.assertEqual(result.missing_by_column["Age"], 177)
        self.assertEqual(result.missing_by_column["Cabin"], 687)
        self.assertEqual(result.missing_by_column["Embarked"], 2)
        for key, age in {1: 38.233440860215055, 2: 29.87763005780347, 3: 25.14061971830986}.items():
            self.assertAlmostEqual(result.mean_age_by_pclass[key], age)
        for key, rate in {1: 136/216, 2: 87/184, 3: 119/491}.items():
            self.assertAlmostEqual(result.survival_rate_by_pclass[key], rate)
        self.assertEqual(result.highest_fares, [512.3292, 512.3292, 512.3292, 263, 263])
        self.assertEqual(path.read_bytes(), original)
