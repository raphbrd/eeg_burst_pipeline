""" Main script managing the workflow for the burst analysis.

@author R. Bordas

Pipeline for burst detection
----------------------------
The cycle-by-cycle analysis of any frequency band burstiness activity is performed in 2 steps:

i) computation of the cycles characteristics, i.e. segmenting the signal into cycles and computing the 
main features of those cycles.

ii) statistics on the cycles characteristics, depending on different configuration: channel-wise,
on a given cluster, windowed data, etc.

How to use it on your own data
------------------------------
1. Update the paths to the outputs directories (*_dir and *_path variables).
2. Update the way to get the list of recordings (`raw_data_paths`) and their associated `recording_ids`.
3. Ensure that the sampling frequency matches the one present in your raw data.
4. (if necessary) update the list of channels, both the full list and the cluster.
5. Run the steps in the given order, as step 2 and step 3 depends on step 1 and step 4 depends on step 2.

Notes
-----
- The input data of this pipeline should be continuous resting-state EEG recordings, in the format .fif. The files will
be loaded through the `mne.io.read_raw_fif` function and thus should be compatible with the mne.io.Raw API.
- This pipeline assumes all data were preprocessed and ready to analyze.
- Before passing the signals to the bycycle algorithm, this pipeline performs two additional preprocessing:
    - filtering according to the parameters specified in the `burst_config.py` file.
    - (as specified in the config) rejecting samples marked with an annotation starting by BAD_*
"""
# %% Configuration
import datetime
import os
import os.path as op

from burst_computation import run_by_cycle_pipeline, run_win_by_cycle_pipeline
from burst_statistics import bursts_stats_all_recordings, run_statistics_pipeline
from helpers import check_pipeline_params, key_ordering_pids
import burst_config as bc

# frequency band name
# this selects the associated parameters in the burst_config.py file
band = 'theta'

#### /!\ Replace the following part by your own way these paths
# - a list of significant channels
root_path = "/Volumes/SSD_NS/2021_RS_studies/RSRT"
output_path: str = op.join(root_path, "results", "bursts", band)
# path to save the output dataframes from bycycle:
cycles_dir: str = op.join(output_path, "cycles_detection")
# path to save the participant-level and channel-level statistics onto those dataframes:
stats_dir: str = op.join(output_path, "cycles_stats")
####

# sampling frequency
sfreq: float = 1000

all_chs: list = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
    'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2',
    'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'AF7',
    'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4',
    'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6',
    'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7',
    'PO8', 'Oz'
]
# the following list of channels is assumed to be the alpha cluster resulting
# from a permutation t test (not included in this pipeline)
significant_chs: list = [
    'T7', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'P5', 'P1',
    'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'TP7', 'TP8', 'PO7', 'PO8',
    'Oz'
]
print(f"Cluster of {len(significant_chs)} significant channels: {str(significant_chs)}")

#### /!\ Replace the following part by your own way to get the recording list
from _local_config import recordings

# this is used to identify any given recording
# e.g., [..., "subjectID_runID", ...]
recording_ids: list = [f"{s}_{b}" for s, b in recordings]
# full path the mne.io.Raw data
raw_data_paths: list = [op.join(root_path, "output_data", s, f"{s}_{b}-inspected-raw.fif") for s, b in recordings]

# the `key` argument for sorting the recordings (passed to `sorted()` or equivalent)
# can be None if not needed
sort_key = key_ordering_pids
####

check_pipeline_params(sfreq, all_chs, significant_chs, raw_data_paths)

# %% [STEP 0] INIT
if not op.exists(output_path):
    os.mkdir(output_path)
    print(f"Output directory {output_path} created.")

# %% [STEP 1] Running cycle-by-cycle analysis on all the sensors
print(f"Running step 1 of time domain analysis with N = {len(recording_ids)} recordings in the {band} frequency band.")

# computing cycles characteristics on all sensors for every block. those data can be filtered
# afterward in function of the desired analysis
run_by_cycle_pipeline(
    recording_ids,
    raw_data_paths=raw_data_paths,
    picks=all_chs,
    band_name=band,
    cycles_dir=cycles_dir,
    use_parallel=True,
)
print(f"Step 1 done.")
now = datetime.datetime.now()
out = f"""-------------- Burst analysis --------------
Datetime:                           \t {now.strftime('%Y-%m-%d %H:%M:%S')}
Number of recordings:                     \t {len(recording_ids)}
Channels:                           \t {all_chs}
Frequency band:                     \t {band}: {bc.freq_bands[band]}
Pre-filter params:                  \t freq. range: {bc.f_filters[band]}, other params: {bc.kwargs_filters[band]}
Burst frequency range:              \t {bc.freqs_burst_range[band]}
Neighboring correction:             \t {bc.use_neighboring_correction[band]}
\t- Mean period correction:         \t {bc.use_mean_period_correction[band]}
Burst detection parameters:
\t- amp_fraction_threshold:         \t {bc.threshold_kwargs[band]["amp_fraction_threshold"]}
\t- amp_consistency_threshold:      \t {bc.threshold_kwargs[band]["amp_consistency_threshold"]}
\t- period_consistency_threshold:   \t {bc.threshold_kwargs[band]["period_consistency_threshold"]}
\t- monotonicity_threshold:         \t {bc.threshold_kwargs[band]["monotonicity_threshold"]}
\t- min_n_cycles:                   \t {bc.threshold_kwargs[band]["min_n_cycles"]}
--------------------------------------------
"""
print(out)
with open(op.join(output_path, f"{band}_burst_detection_report.txt"), "w") as f:
    f.write(out)

# %% [STEP 2] Computing statistics on the cycles features
print(f"Running step 2 of time domain analysis with N = {len(recording_ids)} jobs in the {band} frequency band.")

# computing the statistics for all channels and then only the cluster channels
# this is done for both the average and detailed statistics
# the following loops are redundant but allow easy generation of csv files as outputs
for ch_range_name, ch_range_list in zip(["alpha_cluster", "all_chs"], [significant_chs, all_chs]):
    run_statistics_pipeline(
        recording_ids,
        sfreq,
        picks=ch_range_list,
        file_suffix=f'_{ch_range_name}',
        cycles_dir=cycles_dir,
        stats_dir=stats_dir,
    )
    for stat_level in ["avg", "detailed"]:
        bursts_stats_all_recordings(
            stats_dir,
            [f"{rec_id}_bursts_stats_{stat_level}_{ch_range_name}.csv" for rec_id in recording_ids],
            output_path=output_path,
            output_fname=f"stats_{band}_burst_{stat_level}_{ch_range_name}.csv",
            sort_key=sort_key
        )
print(f"Step 2 done.")

# %% [STEP 3] Running burst analysis on windowed data (30s window)

# computing the cycles characteristics on windowed data. To save computation time, it will simply use
# the cycles computed in step 1 and filter them in function of the windowed data
# as a side effect, overlapping bursts between segments are not taken into account
run_win_by_cycle_pipeline(
    recording_ids,
    sfreq,
    picks=all_chs,
    cycles_dir=cycles_dir,
    win_cycles_dir=cycles_dir,
    use_parallel=True,
    window=30
)
print(f"Step 3 done.")

# %% [STEP 4] Computing statistics on the cycles features for windowed analysis (30s window)
for ch_range_name, ch_range_list in zip(["alpha_cluster", "all_chs"], [significant_chs, all_chs]):
    run_statistics_pipeline(
        recording_ids,
        sfreq,
        picks=ch_range_list,
        window=30,
        file_suffix=f'_{ch_range_name}',
        cycles_dir=cycles_dir,
        stats_dir=stats_dir,
    )
    for stat_level in ["avg", "detailed"]:
        bursts_stats_all_recordings(
            stats_dir,
            [f"{rec_id}_window_30s_bursts_stats_{stat_level}_{ch_range_name}.csv" for rec_id in recording_ids],
            output_path=output_path,
            output_fname=f"stats_windowed_30s_{band}_burst_{stat_level}_{ch_range_name}.csv",
            sort_key=sort_key
        )
print(f"Step 4 done.")
