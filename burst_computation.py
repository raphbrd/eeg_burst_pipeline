""" Main script to compute the bursts and cycles features of the dataset

@author: R. Bordas

Input files:
-----------
- <subject_io>/<run_id>-inspected-raw.fif  # from preprocessing pipeline

Output files:
-----------
- <cycles_dir>/<run_id>_all_cycles.csv  # characteristics for each cycle for all channels
"""
from __future__ import annotations

import os

import mne
import numpy as np
import pandas as pd
from bycycle.burst.utils import check_min_burst_cycles
from bycycle.group import compute_features_2d
from mne.parallel import parallel_func

import burst_config as bc


def get_burst_boundaries(is_burst):
    """ Return the indexes of the first and last cycles of every burst

    This is a streamlined version of neurodsp.burst.utils.compute_burst_stats

    Use the is_burst column of bycycle results. A burst will be a continuous series of True booleans.

    Parameters
    ----------
    is_burst: pd.Series or list of bool
        the column of the dataframe features returned by cycle.

    Return
    ------
    starts: np.ndarray
        the indexes of the first cycle of every burst
    ends: np.ndarray
        the indexes of the last cycle of every burst

    Notes
    -----
    This is how are defined indexes of starts and ends:

        Burst:      =====   =======
        Sequence: 0 1 1 1 0 1 1 1 1 0
        starts      +       +
        ends              *         *

    """
    if not isinstance(is_burst, np.ndarray):
        is_burst = np.array(is_burst, dtype=bool)

    change = np.diff(is_burst)
    idxes, = change.nonzero()

    idxes += 1  # Get indices following the change.

    if is_burst[0]:
        # If the first sample is part of a burst, prepend a zero.
        idxes = np.r_[0, idxes]

    if is_burst[-1]:
        # If the last sample is part of a burst, append an index corresponding
        # to the length of signal.
        idxes = np.r_[idxes, is_burst.size]

    starts = idxes[0::2]
    ends = idxes[1::2]

    return starts, ends


def check_period_in_freq_range(period_array, fs, fmin, fmax):
    """ Check that the periods are in the interval delimited by fmin and fmax. Period should be given in samples """
    return ((fs / period_array) > fmin) & ((fs / period_array) < fmax)


def check_burst_mean_period(df: pd.DataFrame, cycle_idx: int, fs: float, freqs_burst_range: tuple[float, float],
                            avg_func=pd.Series.mean) -> bool:
    """Checks if the mean period of the current burst is within the desired frequency range."""

    if avg_func not in [pd.Series.mean, np.mean, pd.Series.median, np.median]:
        raise ValueError("The average function must be a mean or a median function from pands.Series or numpy.")

    # locating the cycle in a burst
    starts, ends = get_burst_boundaries(df["is_burst"])
    # the cycle is necessarily inside a burst, not at the beginning or the end (by definition)
    current_burst_start = starts[(cycle_idx > starts)][-1]
    current_burst_end = ends[(cycle_idx < ends)][0]
    # cycle at the index 'current_burst_end' is not included in the burst (as it is the one
    # following the last cycle of the burst)
    periods = df.loc[current_burst_start:current_burst_end, "period"]

    # checking if the mean period of the burst is in the burst range
    return check_period_in_freq_range(avg_func(periods), fs, *freqs_burst_range)


def bursts_detection(sigs: np.ndarray,
                     fs: float,
                     picks: list | np.ndarray,
                     freq_band: tuple[float, float],
                     freq_burst_range: tuple[float, float],
                     threshold_kwargs: dict,
                     use_neighboring_correction: bool = True,
                     use_mean_period_correction: bool = True,
                     n_jobs=1) -> pd.DataFrame:
    """ Main function to detect bursts in continuous data.

    Parameters
    ----------
    sigs: np.ndarray
        The signals to detect bursts in. It should be of shape (n_channels, n_samples).
    fs: float
        The sampling frequency of the signal.
    picks: list | np.ndarray
        The list of channels to pick for analysis. This should be identical to the channels used to fetch the data from
        the raw file (e.g., when using raw.get_data(...)).
    freq_band : tuple of float
        The frequency band of interest for the burst detection. Used to filter the signals by bycycle algorithm to
        detect the extrema in the signals.
    freq_burst_range : tuple of float
        The frequency range of the bursts. Used to filter out the bursts that are outside this range. If None, no
        filtering is performed. It can be different from the freq_bands to be more or less conservative on the cycle's
        classification.
    threshold_kwargs : dict
        The parameters of the bycycle algorithm to pass to the `bycycle.features.compute_features` function.
    use_neighboring_correction : bool
        Whether to correct the classification of the cycles based on the period of the neighboring cycles. Only used
        if freqs_burst_range is not None.
    use_mean_period_correction : bool
        Whether to correct the classification of the cycles based on the mean period of the overall burst. Only used
        if freqs_burst_range is not None and use_neighboring_correction is True.
    n_jobs : int
        The number of jobs to run in parallel in the `bycycle.features.compute_features` function.

    Returns
    -------
    The dataframe containing the cycles features and the bursts detection results (`is_burst` column).
    """
    # checking that the number of channels and signals match
    if len(picks) != sigs.shape[0]:
        raise ValueError(f"Number of channels ({len(picks)}) and number of signals ({sigs.shape[0]}) do not match.")

    if use_mean_period_correction and not use_neighboring_correction:
        raise ValueError("The mean period correction can only be used if the neighboring correction is enabled.")

    # parameters from the config files
    compute_kwargs = {
        'burst_method': 'cycles',
        'threshold_kwargs': threshold_kwargs
    }
    # here the default axis = 0 is used, because we compute over independent signals of shape (n_channels, n_samples)
    dfs = compute_features_2d(sigs, fs, freq_band, compute_kwargs, return_samples=True, n_jobs=n_jobs,
                              progress="tqdm" if n_jobs > 1 else None)

    # checking cycle by cycle for each channel
    for df_idx, df in enumerate(dfs):
        # filter out cycles with a period out of frequency range
        df["bursty_features"] = df["is_burst"]  # whether the cycle has bursty characteristics
        df['sensor'] = picks[df_idx]
        if freq_burst_range is not None:
            mask = check_period_in_freq_range(df["period"], fs, *freq_burst_range)
            df['original_burst'] = df['is_burst'] & mask  # cycles that are bursty and in the frequency range
            df['has_correct_period'] = mask

            if use_neighboring_correction:
                for idx, cyc in df[1:-1].iterrows():
                    # if the burst is outside the frequency range, can it be still kept?
                    if cyc["bursty_features"] and not cyc["has_correct_period"]:
                        df.loc[idx, "is_burst"] = False  # by default, it is removed

                        # the cycle might still be considered as a burst if:
                        # i) the cycles right before and after are bursty
                        # ii) the mean period of the current burst is in the burst range
                        if df["is_burst"].loc[idx - 1:idx + 1:2].all():
                            df.loc[idx, "is_burst"] = True

                            if use_mean_period_correction:
                                df.loc[idx, "is_burst"] = check_burst_mean_period(df, idx, fs, freq_burst_range)
            else:
                df["is_burst"] = df["original_burst"]

        # filter out the burst shorter than min_n_cycles after bad periods removal
        min_n_cycles = compute_kwargs["threshold_kwargs"]["min_n_cycles"]
        df['is_burst'] = check_min_burst_cycles(df['is_burst'], min_n_cycles=min_n_cycles)
        cols = df.columns.tolist()
        cols.remove("sensor")
        dfs[df_idx] = df[["sensor"] + cols]

    return pd.concat(dfs)


def cycle_by_cycle_single_rec(data_path, picks, band_name, sid_key, cycles_dir):
    """ Run the burst detection on a single subject / block combination and save the cycles in a csv file.

    Parameters
    ----------
    data_path: str
        The path to the continuous mne.io.Raw data file. The file should be in fif format.
    picks: list
        The channels to keep for the analysis. Should be a list of strings (channel names).
    band_name: str
        the name of the band used to choose the parameters for the burst detection from the burst_config.py file
    sid_key: str
        A unique identifier to the run (e.g., subject_ID_block_ID) to save the results.
    cycles_dir: str
        The path to the directory where to save the results

    Input files
    -----------
    - data_path # continuous raw data

    Output files
    ------------
    - <cycles_dir>/<sid_key>_all_cycles.csv
    """
    raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)
    raw.filter(bc.f_filters[band_name][0], bc.f_filters[band_name][1], verbose=False, **bc.kwargs_filters[band_name])
    # to get proper tracking of ch_names, picks should remain the same:
    sigs = raw.get_data(reject_by_annotation=bc.rejection, picks=picks)

    df_cycles = bursts_detection(
        sigs, raw.info["sfreq"], picks,
        freq_band=bc.freq_bands[band_name],
        freq_burst_range=bc.freqs_burst_range[band_name],
        threshold_kwargs=bc.threshold_kwargs[band_name],
        use_neighboring_correction=bc.use_neighboring_correction[band_name],
        use_mean_period_correction=bc.use_mean_period_correction[band_name]
    )
    df_cycles.to_csv(f"{cycles_dir}/{sid_key}_all_cycles.csv", index=False)


def win_cycle_by_cycle_single_rec(sid_key, window, fs, picks, whole_cycles_dir, win_cycles_dir):
    """ Run the burst detection on the subject `s` and block `b` over segmented data (by bouts of duration `window`)

    The computation over the whole data is done in the function `cycle_by_cycle_single_rec` and is used as a reference.
    The output dataframe is sliced window by window and saved in a csv file.

    Parameters
    ----------
    sid_key: str
        A unique identifier to the run (e.g., subject_ID_block_I, or more simply the NIP) to save the results.
    window: int
        The duration of the window in seconds. Should be an int
    fs: float
        The sampling frequency of the data
    picks: list
    whole_cycles_dir: str
        The path to the directory where the results of the whole data analysis are saved and will be used
        to segment the cycles
    win_cycles_dir: str
        The path to the directory where to save the results

    Input files
    -----------
    - <whole_cycles_dir>/<sid_key>_all_cycles.csv

    Output files
    ------------
    - <win_cycles_dir>/<sid_key>_window_<window>s_all_cycles.csv
    """
    df_cycles = pd.read_csv(f"{whole_cycles_dir}/{sid_key}_all_cycles.csv")
    # we don't need the actual whole duration, as we are not going further in time that the last cycle
    # so only the samples between the last cycle and the last samples are lost
    dur_samples = df_cycles.sample_last_trough.values[-1]

    df_cycles = df_cycles[df_cycles.sensor.isin(picks)]  # filtering the desired channels

    window = int(window)  # to avoid leaving a dot in the filename
    win_samples = int(window * fs)  # number of samples in each window
    n_windows = int(dur_samples // win_samples)
    dfs = []

    # by reusing the cycles previously computed, the main drawback is to lose the cycles that are at the edge of
    # the windows. But if we compute the cycles on the segmented data, we will have to deal with the fact that
    # the cropping will lead to even more edge effects.
    for i in range(n_windows):
        # slicing the dataframe to get only the cycles located inside the window
        # by convention the first sample is included and the last one is excluded
        # so no sample is counted twice
        win_mask = (df_cycles.sample_last_trough >= win_samples * i) & (
                df_cycles.sample_last_trough < win_samples * (i + 1))
        df_window = df_cycles[win_mask]
        df_window["window"] = i
        df_window["window_duration"] = window
        df_window["win_start"] = win_samples * i
        df_window["win_stop"] = win_samples * (i + 1)
        dfs.append(df_window)

    df_windowed = pd.concat(dfs)
    df_windowed["NIP"] = sid_key

    # reordering the columns
    cols = ['NIP', 'sensor', 'window', 'window_duration', 'win_start', 'win_stop',
            'amp_fraction', 'amp_consistency', 'period_consistency', 'monotonicity',
            'period', 'time_peak', 'time_trough', 'volt_peak', 'volt_trough',
            'time_decay', 'time_rise', 'volt_decay', 'volt_rise', 'volt_amp',
            'time_rdsym', 'time_ptsym', 'band_amp', 'sample_peak',
            'sample_last_zerox_decay', 'sample_zerox_decay', 'sample_zerox_rise',
            'sample_last_trough', 'sample_next_trough']
    if 'original_burst' in df_windowed.columns:
        cols.append('original_burst')
    if 'bursty_features' in df_windowed.columns:
        cols.append('bursty_features')
    if 'has_correct_period' in df_windowed.columns:
        cols.append('has_correct_period')
    cols.append('is_burst')
    df_windowed = df_windowed[cols]

    df_windowed.to_csv(f"{win_cycles_dir}/{sid_key}_window_{window}s_all_cycles.csv", index=False)


def run_by_cycle_pipeline(run_ids, raw_data_paths, picks, band_name, cycles_dir="./cycles_analysis",
                          use_parallel=False):
    """ Run the burst detection on the subjects/conditions specified and on the picks given.
    Save cycle by cycle data in a csv file for each block (use cycles_dir path)

    This function needs to be run inside the pipeline (use standard format and config)

    Parameters
    ----------
    run_ids: list
        A list of strings to identify each of the raw file (e.g., can be in the format "subjectID_block")
    raw_data_paths: list
        A list of paths to the continuous mne.io.Raw data file. The file should be in fif format.
    picks: list
        The channels to keep for the analysis. Should be a list of strings (channel names).
    band_name: str
        the name of the band used to choose the parameters for the burst detection from the burst_config.py file
    cycles_dir: str
        The path to the directory where to save the results. If it does not exist, it will be created.
    use_parallel: bool
        Whether to use parallelization or not (default: False). If True, requires the whole pipeline to be run with.

    Input files
    -----------
    - <raw_data_path> # continuous raw data

    Output files
    ------------
    For each run_id, it will generate:
    - <cycles_dir>/<run_id>_all_cycles.csv
    """
    if len(run_ids) != len(raw_data_paths):
        raise ValueError("The number of run_ids and raw_data_paths should be the same.")

    # initiating results directory
    if not os.path.exists(cycles_dir):
        os.mkdir(cycles_dir)

    n_jobs = -1 if use_parallel else 1

    parallel, run_func, _ = parallel_func(cycle_by_cycle_single_rec, n_jobs=n_jobs, total=len(raw_data_paths))
    parallel(run_func(
        data_path=data_path,
        picks=picks,
        band_name=band_name,
        sid_key=run_id,
        cycles_dir=cycles_dir
    ) for run_id, data_path in zip(run_ids, raw_data_paths))


def run_win_by_cycle_pipeline(run_ids, fs, picks, cycles_dir="./cycles_analysis",
                              win_cycles_dir="./windowed_cycles_analysis", use_parallel=False, window=30):
    """ Same as run_by_cycle_pipeline but run the burst detection on segmented data (by bouts of duration `window`).
    To save computation time, it reuses the cycles computed over the whole data and slices them window by window.

    Parameters
    ----------
    run_ids: list
        A list of strings to identify each of the raw file (e.g., can be in the format "subjectID_block")
    fs: float
        The sampling frequency of the data
    picks: list
        The channels to keep for the analysis. Should be a list of strings (channel names).
    cycles_dir: str
        the path to the directory where the results of the whole data analysis are saved and will be used
    win_cycles_dir: str
        The path to the directory where to save the results of the windowed analysis. If it does not exist, it will be
        created.
    use_parallel: bool
        Whether to use parallelization or not (default: False). If True, requires the whole pipeline to be run with.
    window: int
        The duration of the window in seconds. Should be an int

    Input files
    -----------
    - <cycles_dir>/<run_id>_all_cycles.csv

    Output files
    ------------
    For each run_id, it will generate:
    - <win_cycles_dir>/<run_id>_window_<window>s_all_cycles.csv
    """
    if not os.path.exists(win_cycles_dir):
        os.mkdir(win_cycles_dir)

    n_jobs = -1 if use_parallel else 1
    parallel, run_func, _ = parallel_func(win_cycle_by_cycle_single_rec, n_jobs=n_jobs, total=len(run_ids))
    parallel(run_func(
        run_id,
        window=window,
        fs=fs,
        picks=picks,
        whole_cycles_dir=cycles_dir,
        win_cycles_dir=win_cycles_dir,
    ) for run_id in run_ids)
