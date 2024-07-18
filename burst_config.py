""" Configuration file for burst analysis."""

# tuned burst threshold parameters used by
# bycycle.features.compute_features
threshold_kwargs = {
    "alpha": {
        'amp_fraction_threshold': 0.2,
        'amp_consistency_threshold': 0.4,
        'period_consistency_threshold': 0.4,
        'monotonicity_threshold': 0.8,
        'min_n_cycles': 3
    },
    "theta": {
        'amp_fraction_threshold': 0.3,
        'amp_consistency_threshold': 0.4,
        'period_consistency_threshold': 0.4,
        'monotonicity_threshold': 0.7,
        'min_n_cycles': 2
    },
    "beta": {
        'amp_fraction_threshold': 0.2,
        'amp_consistency_threshold': 0.5,
        'period_consistency_threshold': 0.5,
        'monotonicity_threshold': 0.9,
        'min_n_cycles': 3
    },
}

# features keep into the final csv of by_cycle script
features_keep = ['volt_amp', 'band_amp', 'period', 'time_rdsym', 'time_ptsym']

# frequency parameters for bycycle (frequency band of interest)
freq_bands = {
    'theta': (3, 7),
    'alpha': (8, 12),
    'beta': (15, 30),
}

# rejection of samples overlaping BAD_* annotations in MNE
# check mne.io.Raw.get_data documentation for the  details:
# https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.get_data 
rejection = "omit"

# frequency band of the filter before passing data to bycycle
f_filters = {
    'theta': (2, 8),
    'alpha': (freq_bands["alpha"][0] - 4, freq_bands["alpha"][1] + 4),
    'beta': (13, 32),
}
kwargs_filters = {
    "theta": {},  # {"l_trans_bandwidth": 1.25, "h_trans_bandwidth": 1.25},
    "alpha": {},
    "beta": {}
}

# frequency range for the bursty cycles (exclusion after bycycle detection)
# set the value of a key to None if no exclusion for a specific frequency band
freqs_burst_range = {
    'theta': (freq_bands['theta'][0], freq_bands['theta'][1]),
    'alpha': (freq_bands['alpha'][0] - .5, freq_bands['alpha'][1] + .5),
    'beta': (freq_bands['beta'][0], freq_bands['beta'][1]),
}

# Correction of the bycycle output : how to filter bursty cycles
# that do not have a period inside the frequency band of interest ?
# neighboring correction means that a cycle outside of range is still considered as bursty if the two neighboring cycles
# are bursty (before and after this cycle
# mean period correction means that the neighboring correction above holds only if the mean period of the cycles of
# the burst containing this cycle is inside the frequency band of interest
use_neighboring_correction = {
    'theta': True,
    'alpha': True,
    'beta': False,
}
use_mean_period_correction = {
    'theta': True,
    'alpha': True,
    'beta': False,
}
