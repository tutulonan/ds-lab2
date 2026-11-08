"""Проверки Пирсона, попарной обработки пропусков и разделения групп."""
import csv
import math
from pathlib import Path
import tempfile
import unittest

from correlation_tasks import analyze_brain_correlations
from grader_contracts.correlation_tasks import BrainDataInput, BrainCorrelationSummary


FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def manual_pearson(x, y):
    """Независимый расчёт коэффициента по определению, без Pandas."""
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(
        sum(a * a for a in dx) * sum(b * b for b in dy)
    )


class BrainCorrelationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "brainsize.txt"

    def analyze(self, rows):
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(["Gender"] + FEATURES + ["MRI_Count"])
            writer.writerows(rows)
        return analyze_brain_correlations(BrainDataInput(str(self.path)))

    def test_groups_and_correlations_against_formula(self):
        rows = [
            ["Female", 1, 4, 2, 10, 60, 1],
            ["Male", 1, 3, 4, 20, 70, 4],
            ["Female", 3, 3, 5, 30, 64, 2],
            ["Male", 4, 1, 2, 25, 73, 2],
            ["Female", 2, 1, 1, 20, 61, 3],
            ["Male", 2, 2, 3, 30, 71, 3],
            ["Female", 4, 2, 3, 40, 65, 4],
        ]
        result = self.analyze(rows)
        self.assertIsInstance(result, BrainCorrelationSummary)
        self.assertEqual(result.women_count, 4)
        self.assertEqual(result.men_count, 3)
        for gender, actual in [("Female", result.women_mri_correlation),
                               ("Male", result.men_mri_correlation)]:
            group = [row for row in rows if row[0] == gender]
            self.assertEqual(set(actual), set(FEATURES))
            for i, feature in enumerate(FEATURES, 1):
                expected = manual_pearson([row[i] for row in group], [row[-1] for row in group])
                self.assertAlmostEqual(actual[feature], expected)

    def test_na_values_are_removed_pairwise(self):
        rows = [
            ["Male", 1, 4, 2, 10, "NA", 1],
            ["Male", 2, 2, 3, "NA", 70, 2],
            ["Male", 4, 1, 1, 40, 71, 3],
            ["Male", 3, 3, 4, 20, 73, 4],
            ["Male", 9, 9, 9, 99, 99, "NA"],
        ]
        result = self.analyze(rows)
        self.assertEqual(result.men_count, 5)
        self.assertAlmostEqual(result.men_mri_correlation["FSIQ"],
                               manual_pearson([1, 2, 4, 3], [1, 2, 3, 4]))
        self.assertAlmostEqual(result.men_mri_correlation["Weight"],
                               manual_pearson([10, 40, 20], [1, 3, 4]))
        self.assertAlmostEqual(result.men_mri_correlation["Height"],
                               manual_pearson([70, 71, 73], [2, 3, 4]))

    def test_strongest_uses_absolute_value_and_both_groups(self):
        # По очереди самая сильная связь находится в каждой из двух групп.
        for strongest_group in ["Female", "Male"]:
            with self.subTest(group=strongest_group):
                patterns = [[1, 4, 2, 5, 3], [3, 1, 4, 2, 5], [5, 1, 4, 2, 3],
                            [2, 5, 1, 4, 3], [4, 2, 5, 1, 3]]
                rows = []
                for gender in ["Female", "Male"]:
                    for i in range(5):
                        values = [pattern[i] for pattern in patterns]
                        if gender == strongest_group:
                            values[4] = 5 - i
                        rows.append([gender] + values + [i])
                result = self.analyze(rows)
                self.assertEqual(result.strongest_mri_feature, "Height")
                actual = (result.women_mri_correlation if strongest_group == "Female"
                          else result.men_mri_correlation)
                self.assertAlmostEqual(actual["Height"], -1)

    def test_constant_features_are_undefined_and_not_selected(self):
        result = self.analyze([
            ["Female", 7, 7, 3, 7, 7, 1],
            ["Female", 7, 7, 2, 7, 7, 2],
            ["Female", 7, 7, 1, 7, 7, 3],
        ])
        self.assertTrue(math.isnan(result.women_mri_correlation["FSIQ"]))
        self.assertEqual(result.strongest_mri_feature, "PIQ")
        self.assertNotIn("MRI_Count", result.women_mri_correlation)
        self.assertNotIn("MRI_Count", result.men_mri_correlation)

    def test_single_observation_and_empty_group(self):
        result = self.analyze([["Male", 1, 2, 3, 4, 5, 6]])
        self.assertEqual(result.men_count, 1)
        self.assertEqual(result.women_count, 0)
        self.assertTrue(all(math.isnan(value) for value in result.men_mri_correlation.values()))
        self.assertTrue(all(math.isnan(value) for value in result.women_mri_correlation.values()))
        self.assertEqual(result.strongest_mri_feature, "")

    def test_constant_mri_has_no_correlation(self):
        result = self.analyze([["Female", i, i, i, i, i, 100] for i in range(3)])
        self.assertTrue(all(math.isnan(value) for value in result.women_mri_correlation.values()))
        self.assertEqual(result.strongest_mri_feature, "")

    def test_empty_table(self):
        result = self.analyze([])
        self.assertEqual(result.men_count, 0)
        self.assertEqual(result.women_count, 0)
        self.assertEqual(result.strongest_mri_feature, "")

    def test_attached_dataset(self):
        path = Path(__file__).resolve().parents[1] / "brainsize.txt"
        original = path.read_bytes()
        result = analyze_brain_correlations(BrainDataInput(str(path)))
        self.assertEqual(result.men_count, 20)
        self.assertEqual(result.women_count, 20)
        self.assertEqual(result.strongest_mri_feature, "PIQ")
        expected_women = {"FSIQ": .325697, "VIQ": .254933, "PIQ": .396157,
                          "Weight": .446271, "Height": .174541}
        expected_men = {"FSIQ": .498369, "VIQ": .413105, "PIQ": .568237,
                        "Weight": -.076875, "Height": .301543}
        for expected, actual in [(expected_women, result.women_mri_correlation),
                                 (expected_men, result.men_mri_correlation)]:
            for feature, value in expected.items():
                self.assertAlmostEqual(actual[feature], value, places=6)
        self.assertEqual(path.read_bytes(), original)
