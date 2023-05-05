#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on May 05 2023

@author: Price Withers, Kayla Togneri
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import retrieve_track_run, shuffle_nt_arr

neu_dir = Path("/Volumes/shares/NEU")
dwi_dir = neu_dir / 'Projects' / 'DWI'
meg_dir = neu_dir / 'Projects' / 'CTF_MEG'


if __name__ == "__main__":

    # parse arguments
    purpose = ""
    parser = ArgumentParser(description=purpose)
    parser.add_argument("--pnum", help="subject p-number")
    parser.add_argument("--parcs", help='num of schaefer parcels')

    args = parser.parse_args()
    pnum = args.pnum
    n_parcs = int(args.parcs)

    subj_dwi_dir = dwi_dir / pnum
    subj_meg_dir = meg_dir / pnum

    prob_dir = subj_dwi_dir / 'archive' / 'track' / 'prob'
    log_path = prob_dir / 'track_log'
    run_num = retrieve_track_run(
        log_file=log_path,
        n_parcs=n_parcs
    )

    all_data_path = prob_dir / run_num / 'csv' / 'all_data.npy'
    all_data = np.load(
        file=all_data_path,
        allow_pickle=True
    ).item()

    nt_arr = all_data['NT']
    nt_arr[np.diag_indices_from(nt_arr)] = 0

    # TODO: loop through all available runs
    delay_path = subj_meg_dir / 'dSPM' / 'run_07' / 'propagation_delays.csv'
    delays = np.loadtxt(
        fname=delay_path, 
        delimiter=','
    )

    source_parc_idx = np.argwhere(delays == 0).squeeze()
    follower_parc_idxs = np.argwhere(delays > 0).squeeze()
    follower_lags = delays[delays > 0]

    nt_followers = nt_arr[source_parc_idx,follower_parc_idxs]
    total_nt = np.sum(nt_followers)

    random_totals = shuffle_nt_arr(
        nt_arr.copy(),
        nt_followers.size
    )

    print("DEBUG")
