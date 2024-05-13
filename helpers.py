""" Miscellaneous helper functions """
import pandas as pd
import mne
import re


def check_ch_exists(channel, chs_list):
    """ Check if the channel exists in a given list. """
    if isinstance(channel, str):
        channel = [channel]

    for s in channel:
        if s not in chs_list:
            raise ValueError(f"{s} is not in the given channel names")


def concat_csv_files(fnames_list):
    """ Concatenate a list of csv files from a list of file names """
    dfs_list = list()
    for csv_file in fnames_list:
        csv = pd.read_csv(csv_file)
        dfs_list.append(csv)

    df_concat = pd.concat(dfs_list)
    return df_concat


def check_pipeline_params(sfreq, all_chs, significant_chs, raw_data_paths):
    """ Check that the parameters manually specified in the scripts match the ones 
    internally detected by MNE. """
    if len(raw_data_paths) < 1:
        raise ValueError("There must be at list one recording as input data of the pipeline")

    info = mne.io.read_info(raw_data_paths[0], verbose=False)

    if sfreq != info["sfreq"]:
        raise ValueError("The sampling frequency specified in the script does not match "
                         "the one detected by MNE in the raw data")

    check_ch_exists(all_chs, info["ch_names"])
    check_ch_exists(significant_chs, info["ch_names"])


def key_ordering_pids(x):
    """ Return the key to order a list of strings by the numerical values of digits in the 
    format *_(\d+), e.g., P_21 will come after P_3 (contrary to the internal python string ordering) """
    search = re.search(r'_(\d+)', x)
    if search:
        return int(search.group(1))
    return 0
