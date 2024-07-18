""" Unit testing of key helper functions used in the burst detection pipeline.

Usage
-----
- Run the test suite by running `python -m unittest test_burst_detection`
"""
import unittest

import numpy as np
import pandas as pd

from burst_computation import get_burst_boundaries, check_period_in_freq_range, check_burst_mean_period
from burst_statistics import compute_cycles_statistics


class BurstComputationTestCases(unittest.TestCase):

    def setUp(self):
        self.burst = np.zeros(10, dtype=bool)
        self.burst[1:4] = 1  # 1 burst starting at idx 1 and ending at idx 3
        self.burst[5:9] = 1  # 1 burst starting at idx 5 and ending at idx 8

    def test_regular_burst_boundaries(self):
        starts, ends = get_burst_boundaries(self.burst)
        np.testing.assert_array_equal(starts, np.array([1, 5]))
        np.testing.assert_array_equal(ends, np.array([4, 9]))

    def test_regular_burst_boundaries_with_floats(self):
        burst = np.zeros(10, dtype=float)  # 1 or 0 float dtypes
        burst[1:4] = 1  # 1 burst starting at idx 1 and ending at idx 3
        burst[5:9] = 1  # 1 burst starting at idx 5 and ending at idx 8

        starts, ends = get_burst_boundaries(list(burst))
        np.testing.assert_array_equal(starts, np.array([1, 5]))
        np.testing.assert_array_equal(ends, np.array([4, 9]))

    def test_first_sample_burst_boundaries(self):
        burst = np.zeros(10, dtype=bool)
        burst[0:4] = 1  # 1 burst starting at idx 1 and ending at idx 3
        starts, ends = get_burst_boundaries(burst)
        np.testing.assert_array_equal(starts, np.array([0]))
        np.testing.assert_array_equal(ends, np.array([4]))

    def test_last_sample_burst_boundaries(self):
        burst = np.zeros(10)
        burst[5:10] = 1  # 1 burst starting at idx 5 and ending at idx 8

        starts, ends = get_burst_boundaries(burst)
        np.testing.assert_array_equal(starts, np.array([5]))
        np.testing.assert_array_equal(ends, np.array([10]))

    def test_check_period_range(self):
        periods = np.array([100, 70, 140, 120, 30, 40])
        fs = 1000
        np.testing.assert_array_equal(
            check_period_in_freq_range(periods, fs, 8, 12),
            [True, False, False, True, False, False])

        periods = np.array([100, 70, 140, 120, 30, 40])
        fs = 500
        np.testing.assert_array_equal(
            check_period_in_freq_range(periods, fs, 8, 12),
            [False, False, False, False, False, False])


class BurstPeriodChecksTestCases(unittest.TestCase):

    def setUp(self):
        self.bursts = np.zeros(20, dtype=bool)  # 1 burst of 3 cycles and 1 burst of 5 cycles
        self.bursts[1:4] = 1
        self.bursts[10:15] = 1
        self.periods = np.repeat([100, 110, 90, 120, 95], 4)

        self.df_cycles = pd.DataFrame({
            "sensor": ["sensor1"] * 20,
            "is_burst": self.bursts,
            "period": self.periods,
        })

    def test_period_of_a_burst_cycle_ok(self):
        self.assertTrue(check_burst_mean_period(self.df_cycles, 2, 1000, (8, 12)))

    def test_period_of_a_burst_cycle_not_ok(self):
        tmp = self.df_cycles.copy()
        tmp.loc[1, "period"] = 60
        tmp.loc[3, "period"] = 60
        self.assertFalse(check_burst_mean_period(tmp, 2, 1000, (8, 12)))

    def test_period_of_a_start_burst_cycle(self):
        # testing period of a boundary cycle
        self.assertTrue(
            check_burst_mean_period(
                self.df_cycles, 1, 1000, (8, 12))
        )

    def test_period_of_an_end_burst_cycle(self):
        # testing period of a boundary cycle
        self.assertTrue(
            check_burst_mean_period(
                self.df_cycles, 1, 1000, (8, 12))
        )
        self.assertTrue(
            check_burst_mean_period(
                self.df_cycles, 3, 1000, (8, 12))
        )

        # testing period of the end cycle (by definition, it is outside of the burst)
        self.assertRaises(
            ValueError,
            check_burst_mean_period,
            self.df_cycles, 4, 1000, (8, 12)
        )

    def test_trying_to_check_period_of_a_non_burst_cycle(self):
        # testing period of a non-burst cycle
        self.assertRaises(
            ValueError,
            check_burst_mean_period,
            self.df_cycles, 0, 1000, (8, 12)
        )

        self.assertRaises(
            ValueError,
            check_burst_mean_period,
            self.df_cycles, 7, 1000, (8, 12)
        )

    def test_not_using_a_valid_avg_func(self):
        # testing period of a non-burst cycle
        self.assertRaises(
            ValueError,
            check_burst_mean_period,
            self.df_cycles, 0, 1000, (8, 12), avg_func="invalid"
        )


class BurstStatisticsTestCases(unittest.TestCase):

    def setUp(self):
        bursts1 = np.zeros(20, dtype=bool)  # 1 burst of 3 cycles and 1 burst of 5 cycles
        bursts1[1:4] = 1
        bursts1[10:15] = 1
        bursts2 = np.zeros(20, dtype=bool)  # 2 bursts of 5 cycles
        bursts2[6:11] = 1
        bursts2[13:18] = 1
        periods1 = np.repeat([100, 110, 90, 120, 95], 4)
        periods2 = np.repeat([95, 105, 97, 103, 110], 4)

        self.bursts = np.concatenate([bursts1, bursts2])
        self.periods = np.concatenate([periods1, periods2])

        self.overlap_bursts = np.concatenate([bursts1[0:10], bursts1[15:20], bursts2[0:2],
                                              bursts1[10:15], bursts2[2:]])
        self.overlap_periods = np.concatenate([periods1[0:10], periods1[15:20], periods2[0:2],
                                               periods1[10:15], periods2[2:]])

    def test_compute_cycles_statistics_whole_recording(self):
        # simulating an output dataframe of the burst computation pipeline with 20 cycles
        df_cycles = pd.DataFrame({
            "sensor": ["sensor1"] * 20 + ["sensor2"] * 20,
            "is_burst": self.bursts,
            "period": self.periods,
        })

        # computing the statistics
        stats = compute_cycles_statistics(df_cycles, fs=1000, average=False)
        self.assertEqual(stats[stats.sensor == "sensor1"]["n_bursts"].values[0], 2)
        self.assertEqual(stats[stats.sensor == "sensor1"]["n_bursty_cycles"].values[0], 8)
        self.assertEqual(stats[stats.sensor == "sensor1"]["n_cycles_per_burst"].values[0], 4)
        self.assertEqual(stats[stats.sensor == "sensor1"]["inter_burst_dur_mean"].values[0], (100 + 620 + 500) / 3)
        self.assertEqual(stats[stats.sensor == "sensor1"]["burst_dur_mean"].values[0], (300 + 540) / 2)

        stats = compute_cycles_statistics(df_cycles, fs=1000, average=True)
        self.assertEqual(stats["n_bursts"], 2)
        self.assertEqual(stats["n_bursty_cycles"], 9)
        self.assertEqual(stats["n_cycles_per_burst"], 4.5)

    def test_compute_cycles_statistics_windowed_recording(self):
        df_cycles = pd.DataFrame({
            "sensor": ["sensor1"] * 40,
            "window": [0] * 20 + [1] * 20,
            "is_burst": self.bursts,
            "period": self.periods,
        })
        stats0 = compute_cycles_statistics(df_cycles[df_cycles.window == 0], fs=1000, average=False)
        stats1 = compute_cycles_statistics(df_cycles[df_cycles.window == 1], fs=1000, average=False)
        self.assertEqual(stats0["n_bursts"].values[0], 2)
        self.assertEqual(stats0["n_bursty_cycles"].values[0], 8)
        self.assertEqual(stats0["n_cycles_per_burst"].values[0], 4)
        self.assertEqual(stats0["inter_burst_dur_mean"].values[0], (100 + 620 + 500) / 3)
        self.assertEqual(stats0["burst_dur_mean"].values[0], (300 + 540) / 2)
        self.assertEqual(stats1["inter_burst_dur_mean"].values[0], (590 + 200 + 220) / 3)
        self.assertEqual(stats1["burst_dur_mean"].values[0], (501 + 529) / 2)

    def test_compute_cycles_statistics_windowed_overlap_bursts(self):
        """ Test bursts overlap a window boundary."""
        df_cycles = pd.DataFrame({
            "sensor": ["sensor1"] * 40,
            "window": [0] * 20 + [1] * 20,
            "is_burst": self.overlap_bursts,
            "period": self.overlap_periods,
        })
        stats0 = compute_cycles_statistics(df_cycles[df_cycles.window == 0], fs=1000, average=False)
        self.assertEqual(stats0["n_bursts"].values[0], 1)
        self.assertEqual(stats0["n_bursty_cycles"].values[0], 3)
        self.assertEqual(stats0["n_cycles_per_burst"].values[0], 3)
        self.assertEqual(stats0["burst_dur_mean"].values[0], 300)
        self.assertEqual(stats0["inter_burst_dur_mean"].values[0], (100 + 1310) / 2)

        stats1 = compute_cycles_statistics(df_cycles[df_cycles.window == 1], fs=1000, average=False)
        self.assertEqual(stats1["n_bursts"].values[0], 2)
        self.assertEqual(stats1["burst_dur_mean"].values[0], 515)  # (501 + 529) / 2
        self.assertEqual(stats1["inter_burst_dur_mean"].values[0], (400 + 200 + 220) / 3)
        self.assertEqual(stats1["percent_bursty_cycles"].values[0], 10 / 18 * 100)
