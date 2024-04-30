""" Compute statistics on the bursts detected for each block and save them in a csv file.

@author: R. Bordas

Input files:
-----------
- <cycle_fil_path>/<pid>_<block_name>_all_cycles.csv  # from burst_computation.py

Output files:
------------
- <stats_file_path>/<pid>_<block_name>_bursts_stats.csv  # averaged statistics over all channels
- <stats_file_path>/<pid>_<block_name>_bursts_stats_channels.csv  # detailed stats for every channel
"""
import os
from typing import Any

import numpy as np
import pandas as pd
from mne.parallel import parallel_func

import burst_config as bc
from helpers import check_ch_exists, concat_csv_files


def bursts_stats_all_recordings(stats_file_path: str, csv_filenames: list, output_path: str = None,
                                output_fname: str = None, sort_key: Any = None):
    """ Concatenate the stats of all recordings  and save them in a csv file."""
    if output_fname is not None and output_path is None or output_fname is None and output_path is not None:
        raise ValueError("Both output_fname and output_path must be either defined to save the output or undefined at "
                         "the same time to not save it.")

    csv_filenames = [os.path.join(stats_file_path, csv_file) for csv_file in csv_filenames]

    csv_filenames.sort(key=sort_key)
    df_all_subjects = concat_csv_files(csv_filenames)

    if output_fname is not None:
        df_all_subjects.to_csv(os.path.join(output_path, output_fname), index=False)

    return df_all_subjects


def compute_cycles_statistics(df_cycles: pd.DataFrame, fs: float, average: bool = True):
    """ Compute statistics over a data frame of all the cycles of a time series.
    To compute statistics over the samples, we are using the period as the sample length of a cycle.

    Adapted from neurodsp.burst.compute_burst_stats to take into account samples-level statistics and
    not only cycle-level statistics.

    Parameters
    ----------
    df_cycles: pd.DataFrame
        Output of bycycle.features.compute_features, with a column 'sensor'
    fs: float
        sampling frequency of the original time series
    average: bool
        whether to average the statistics over the channels
    window: int
        Whether the statistics are computed over windows or over the whole block. If None (default), assumes that
        the first cycle is not part of a burst. If  an integer, does not make this assumption unless this is the
        first window (i.e. the first cycle of the whole recording).

    Return
    ------
    A dictionary of the averaged cycles statistics, either per channel or for all channels aggregated (average).
    """

    groups = df_cycles.groupby("sensor")
    dfs = []

    for sensor_name, group_df in groups:
        group = group_df.reset_index()
        tot_time = group["period"].sum() / fs
        starts = np.array([])
        ends = np.array([0])  # the time series never start with a burst
        burst_durations = np.array([])
        non_burst_durations = np.array([])

        # because the following assumes that the first and last cycles are not part of a burst
        # they are removed if this is the case. This can happen if using windowed data for example.
        # TODO: this might not be the best way to handle this
        if group["is_burst"].iloc[0]:
            first_inter_burst_start = group[~group["is_burst"]].index[0]
            group = group.iloc[first_inter_burst_start:].reset_index(drop=True)
        if group["is_burst"].iloc[-1]:
            last_inter_burst_end = group[~group["is_burst"]].index[-1] + 1
            group = group.iloc[:last_inter_burst_end].reset_index(drop=True)

        for ii, burst_index in enumerate(np.where(np.diff(group["is_burst"]) != 0)[0]):
            if (ii % 2) == 0:
                starts = np.append(starts, burst_index + 1)
            else:
                # this is the index for the end of the current burst
                ends = np.append(ends, burst_index + 1)
                # remove 1 because Pandas are inclusive on both sides, contrary to Numpy
                current_inter_burst_end = starts[-1] - 1
                current_burst_end = ends[-1] - 1
                burst_durations = np.append(burst_durations,
                                            group.loc[starts[-1]:current_burst_end, "period"].sum())
                non_burst_durations = np.append(non_burst_durations,
                                                group.loc[ends[-2]:current_inter_burst_end, "period"].sum())

        if len(starts) > 1:
            n_cycles_per_burst = (ends[1:] - starts).mean()
            n_cycles_per_burst_var = (ends[1:] - starts).var()
        else:
            n_cycles_per_burst = np.nan
            n_cycles_per_burst_var = np.nan
        non_burst_durations = np.append(non_burst_durations,
                                        group.loc[ends[-1]:group.shape[0] - 1, "period"].sum())
        dfs.append(pd.DataFrame([{
            "sensor": sensor_name,
            # absolute counts
            'n_bursts': len(starts),
            'n_bursty_cycles': group["is_burst"].sum(),
            'n_cycles_per_burst': n_cycles_per_burst,
            'n_cycles_per_burst_var': n_cycles_per_burst_var,

            # bursts and interval between bursts durations in samples
            'inter_burst_dur_mean': np.mean(non_burst_durations) if len(non_burst_durations) > 0 else np.nan,
            'inter_burst_dur_std': np.std(non_burst_durations) if len(non_burst_durations) > 0 else np.nan,
            'burst_dur_mean': np.mean(burst_durations) if len(burst_durations) > 0 else np.nan,
            'burst_dur_std': np.std(burst_durations) if len(burst_durations) > 0 else np.nan,

            # relative indexes
            'percent_bursty_samples': 100 * np.sum(burst_durations) / group["period"].sum(),
            'percent_bursty_cycles': 100 * group["is_burst"].sum() / len(group["is_burst"]),
            'bursts_per_second': len(starts) / tot_time,
        }]))

    if average:
        # TODO : harmonize return type, it should be a dataframe instead of a dict.
        out = pd.concat(dfs)
        return out.drop("sensor", axis=1).mean().to_dict()

    return pd.concat(dfs, ignore_index=True)


def compute_sensor_stats(df_sensor):
    """ Compute statistics for a sensor, for bursty and non-bursty cycles. """
    stats = {}
    for feature in bc.features_keep:
        non_bursty = df_sensor[~df_sensor.is_burst][feature].mean()
        bursty = df_sensor[df_sensor.is_burst][feature].mean()
        stats.update({"non_bursty_" + feature: non_bursty, "bursty_" + feature: bursty})
    return pd.DataFrame(stats, index=[0])


def burst_stats_per_recording(sid_key: str, fs: float, cycle_file_path: str, stats_file_path: str = None,
                              picks=None, file_suffix=""):
    """ Compute statistics on the bursts detected for each recording and save them in a csv file.

    Parameters
    ----------
    sid_key: str
    fs: float
        sampling frequency of the original time series
    cycle_file_path: str
        path to the folder containing the cycles data frame
    stats_file_path: str
        path to the folder where to save the statistics
    picks: list
        channels to keep for the statistics
    file_suffix: str
        suffix to add to the statistics file name

    Input files
    -----------
    - <cycle_file_path>/<sid_key>_all_cycles.csv

    Output files
    ------------
    - <stats_file_path>/<sid_key>_bursts_stats_avg<file_suffix>.csv
    - <stats_file_path>/<sid_key>_bursts_stats_detailed<file_suffix>.csv
    """
    # output of the computation step of the pipeline
    df_cycles = pd.read_csv(f"{cycle_file_path}/{sid_key}_all_cycles.csv")
    if picks is not None:
        check_ch_exists(picks, df_cycles.sensor.unique())
        df_cycles = df_cycles[df_cycles.sensor.isin(picks)]

    stats_bursts = compute_cycles_statistics(df_cycles, fs=fs, average=False)
    stats_features = df_cycles.groupby('sensor').apply(compute_sensor_stats, include_groups=False).reset_index()

    out = pd.merge(stats_bursts, stats_features.drop(["level_1"], axis=1), on="sensor")

    out_mean = out.drop(["sensor"], axis=1).mean()
    out_mean = pd.concat([pd.Series({"NIP": sid_key}), out_mean])
    out_mean = out_mean.to_frame().T
    out.insert(0, "NIP", sid_key)

    # 3. saving everything
    if stats_file_path:
        out_mean.to_csv(f"{stats_file_path}/{sid_key}_bursts_stats_avg{file_suffix}.csv",
                        index=False)
        out.to_csv(f"{stats_file_path}/{sid_key}_bursts_stats_detailed{file_suffix}.csv",
                   index=False)

    return out


def burst_stats_per_window(sid_key: str, fs: float, window: int, cycle_file_path: str,
                           stats_file_path: str = None, picks=None, file_suffix=""):
    """ Compute statistics on the bursts detected for each window of a recording and save them in a csv file.

    Parameters
    ----------
    sid_key: str
    fs: float
        sampling frequency of the original time series
    window: int
        window duration in seconds
    cycle_file_path: str
        path to the folder containing the cycles data frame
    stats_file_path: str
        path to the folder where to save the statistics
    picks: list
        channels to keep for the statistics
    file_suffix: str
        suffix to add to the statistics file name

    Input files
    -----------
    - <cycle_file_path>/<sid_key>_window_<window>s_all_cycles.csv

    Output files
    ------------
    - <stats_file_path>/<sid_key>_window_<window>s_bursts_stats_detailed<file_suffix>.csv
    - <stats_file_path>/<sid_key>_window_<window>s_bursts_stats_avg<file_suffix>.csv
    """
    # output of the computation step of the pipeline
    df_cycles = pd.read_csv(f"{cycle_file_path}/{sid_key}_window_{window}s_all_cycles.csv")
    if picks is not None:
        check_ch_exists(picks, df_cycles.sensor.unique())
        df_cycles = df_cycles[df_cycles.sensor.isin(picks)]

    windows = df_cycles["window"].unique()
    stats_bursts = []
    stats_features = []
    for i in windows:
        df_cycles_win = df_cycles[(df_cycles["window"] == i)]
        stats = compute_cycles_statistics(df_cycles_win, fs=fs, average=False)
        stats["window"] = i
        stats_bursts.append(stats)
        stats = df_cycles_win.groupby('sensor').apply(compute_sensor_stats, include_groups=False).reset_index()
        stats["window"] = i
        stats_features.append(stats)
    stats_bursts_df = pd.concat(stats_bursts)
    stats_features_df = pd.concat(stats_features)
    out = pd.merge(stats_bursts_df, stats_features_df.drop(["level_1"], axis=1), on=["sensor", "window"])
    out.insert(0, "NIP", sid_key)
    out.insert(1, "window_duration", window)

    if stats_file_path:
        fname = f"{stats_file_path}/{sid_key}_window_{window}s_bursts_stats_detailed{file_suffix}.csv"
        out.to_csv(fname, index=False)
        out_mean = out.groupby(by=["window"]).mean(numeric_only=True).reset_index(drop=False)
        fname = f"{stats_file_path}/{sid_key}_window_{window}s_bursts_stats_avg{file_suffix}.csv"
        out_mean.insert(0, "NIP", sid_key)
        out_mean.to_csv(fname, index=False)

    return out


def run_statistics_pipeline(run_ids: list, sfreq: float, picks: list = None, window: int = None,
                            cycles_dir: str = "./cycles_analysis", stats_dir: str = "./cycles_stats",
                            file_suffix: str = ""):
    """ Save overall stats for each block in a csv file (use stats_dir path)

    Parameters
    ----------
    run_ids: list
        A list of tuples (subject, block_name)
    sfreq: float
        sampling frequency of the original time series
    picks: list
        channels to keep for the statistics
    window: int | None
        window duration in seconds to compute statistics over temporal windows. If None, compute statistics over the
        whole block. If not None, the cycles data frames should have been computed with the window parameters.
    cycles_dir: str
        path to the folder containing the cycles data frame (output of burst_computation.py)
    stats_dir: str
        path to the folder where to save the statistics. Will be created if it does not exist
    file_suffix: str
        suffix to add to the statistics file name

    Input files
    -----------
    - <cycles_dir>/<run_id>_all_cycles.csv

    Output files
    ------------
    For each run_id, it will generate:
    - <stats_dir>/<run_id>_bursts_stats_avg<file_suffix>.csv
    - <stats_dir>/<run_id>_bursts_stats_detailed<file_suffix>.csv
     The file with the suffix _channels contains the statistics channel-wise, while the other file contains the
    statistics aggregated over all picks.
    """
    if not os.path.exists(stats_dir):
        os.mkdir(stats_dir)

    if window is None:
        parallel, run_func, _ = parallel_func(burst_stats_per_recording, n_jobs=-1, total=len(run_ids))
        parallel(run_func(
            run_id,
            fs=sfreq,
            cycle_file_path=cycles_dir,
            stats_file_path=stats_dir,
            picks=picks,
            file_suffix=file_suffix
        ) for run_id in run_ids)
    else:
        parallel, run_func, _ = parallel_func(burst_stats_per_window, n_jobs=-1, total=len(run_ids))
        parallel(run_func(
            run_id,
            fs=sfreq,
            cycle_file_path=cycles_dir,
            stats_file_path=stats_dir,
            picks=picks,
            window=window,
            file_suffix=file_suffix
        ) for run_id in run_ids)
