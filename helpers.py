#!/usr/bin/env python

import random

import numpy as np
import pandas as pd

def retrieve_track_run(log_file, n_parcs=400):

    df = pd.read_csv(log_file, delimiter=' ', index_col='track_idx')
    runs_with_parcs = df.loc[df.num_parc == n_parcs].index

    # ensure that only one row has correct number of parcels
    assert runs_with_parcs.size == 1

    run_num = str(runs_with_parcs[0])

    if len(run_num) == 1:
        run_num = '0' + run_num

    return run_num

def shuffle_nt_arr(nt_arr, n_followers, n_shuffles=10000):

    nt_totals = np.full(n_shuffles, np.NaN)

    for shuffle in range(n_shuffles):
        random_source = random.sample(range(nt_arr.shape[0]), k=1)
        possible_followers = [i for i in range(nt_arr.shape[0]) if i != random_source[0]]
        random_followers = random.sample(possible_followers,k=n_followers)

        nt_followers = nt_arr[random_source,random_followers]
        total_nt = np.sum(nt_followers)

        nt_totals[shuffle] = total_nt

    return nt_totals

def reorder_prop_delays(fs_delays, fs_annot, dwi_labels):

    fs_order = [f"{i.hemi.upper()}_{i.name[14:-3]}" for i in fs_annot]
    out_delays = np.full(fs_delays.shape, np.NaN)

    for i, label in enumerate(dwi_labels.label.values):

        fs_label_idx = fs_order.index(label)
        out_delays[i] = fs_delays[fs_label_idx]

    return out_delays