# Burst detection pipeline for scalp EEG data

This pipeline is designed to detect bursts in scalp EEG data for a specific frequency band (e.g., alpha between 8 and 12
Hz).

The main purpose is to replicate with the EEG data the results from [1], namely the prediction of minute-scale estimated
durations (retrospective settings) by alpha burstiness.

Author: R. Bordas

## Pipeline steps

1. Compute the cycles features using the `bycycle.features.compute_features` function. It saves the output in the csv
   format in the `cycles_dir` directory. The `burst_computation.py` contains functions wrapping the cycle-by-cycle
   algorithm [2].

2. Compute channel-level and participant-level statistics over those cycles. The `burst_stats.py` script contains the
   functions to compute the burstiness markers.

The `run_burst_pipeline` is the main entry script and contains some configuration variables to update.

Main outputs of interest:

- **<output_path>/stats_<band_name>_burst_detailed_all_chs.csv**: channel-wise burstiness markers per recording.
- **<output_path>/stats_<band_name>_burst_avg_all_chs.csv**: average per participant across all channels of the
  previous file.
- **<output_path>/stats_<band_name>_burst_detailed_alpha_cluster.csv**: same as the first file, but containing only
  channels from the specified cluster.
- **<output_path>/stats_<band_name>_burst_avg_alpha_cluster.csv**: average per participant across all channels of the
  previous file.

`<output_path>` and `<band_name>`  are specified at the beginning of the main script.

Main burstiness markers (columns names in the output dataframes):

- percent_bursty_cycles: this is the relative burst time from the original study. Defined as
  $\text{burst time} = \frac{\sum c_{i}}{N} \times 100$
  With $N$ the number of cycles in the cycles and $c_i$ the boolean variable specifying if cycle $i$ is part of a burst
  or not.

- bursty_volt_amp: the mean amplitude of the cycles classified as bursty.

- bursty_period: mean period of the cycles classified as bursty.

- burst_dur_mean: averaged duration of a burst.

- inter_burst_dur_mean: averaged duration of the interval between two consecutive bursts.

- n_bursts: total number of bursts.

- bursts_per_second: n_bursts divided by the total duration of the recording.

We denote by 'bursty' a cycle classified as part of a burst (a sequence of contiguous consistent cycles in terms of
period and amplitude).

Note: if using `rejection = "omit"` (see `burst_config.py`file), there is no guarantee the recordings will have the same
duration (because of variable rejection of BAD* segments). In this case, relative values of burstiness (e.g., 
bursts_per_second, percent_bursty_cycles) should be used instead of absolute ones (e.g., n_bursts, n_bursty_cycles...). 

## Installation

### From source

1. Download the repository

```bash
git clone https://github.com/raphbrd/eeg_burst_pipeline
```

2. Install the packages with pip

```bash
pip install -r requirements.txt
```

### Dependencies

See the requirements.txt file for the exhaustive list.

- mne, v1.7.0
- numpy
- pandas
- tqdm
- bycycle, v1.0.0
- neurodsp, v2.2.1

Note: this pipeline is not compatible with the latest version of bycycle (v1.1.0) due to changes in the API.

## Usage

It is assumed that the input data is already preprocessed, consisting of continuous raw EEG data (FIF format used
by MNE-Python). Importantly, data are assumed to be cropped between the start and end of the recording of interest.
The window analysis is meant to crop the data further but to study the stability over time of the oscillation.

1. Configure the `run_burst_pipeline.py` script with the correct paths to the input data and output directories

2. Provide the sampling frequency of the data. It is assumed that all recordings have the same sampling frequency.

3. Provide a cluster of interest (e.g., from a permutation test)

4. Run:

```bash
python run_burst_pipeline.py
```

## Parameters used in our studies

| 	                              | alpha    	 | theta 	 |
|--------------------------------|------------|---------|
| Frequency band (Hz)          	 | 8-12     	 | 4-7   	 |
| amp_fraction_threshold       	 | 0.2      	 | 0.3   	 |
| amp_consistency_threshold    	 | 0.4      	 | 0.4   	 |
| period_consistency_threshold 	 | 0.4      	 | 0.4   	 |
| monotonicity_threshold       	 | 0.8      	 | 0.8   	 |
| min_n_cycles                 	 | 3        	 | 2     	 |
| Filter frequencies (Hz)      	 | 4-16     	 | 2-8   	 |
| Burst range (Hz)             	 | 7.5-12.5 	 | 3-7   	 |

Note: we use the default definition of a cycle in the bycycle package, i.e. the center extrema is the peak of the
cycle.

## References

[1] Azizi, L., Polti, I., & Van Wassenhove, V. (2023). Spontaneous α Brain Dynamics Track the Episodic “When”. The
Journal of Neuroscience, 43(43), 7186‑7197. https://doi.org/10.1523/JNEUROSCI.0816-23.2023

[2] Cole, S., & Voytek, B. (2019). Cycle-by-cycle analysis of neural oscillations. Journal of Neurophysiology, 122(2),
849‑861. https://doi.org/10.1152/jn.00273.2019
