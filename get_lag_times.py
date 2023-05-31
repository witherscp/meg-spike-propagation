#!/usr/bin/python

"""

@author: Kaya Scheman

"""

# import modules
import numpy as np
import sys
from pathlib import Path
import mne

# args

pnum_subj = sys.argv[1]
run_num = sys.argv[2]

# set paths

neu_dir = Path("/Volumes/shares/NEU")
megdir = neu_dir / "Projects/CTF_MEG" 
mri_subj_dir = Path("/Volumes/shares/NEU/Projects/Freesurfer/stable-v6.0.0/Linux/subjects")


# load array 
dspm_dir = megdir / pnum_subj / 'dSPM' / f'run_0{run_num}'
file =  dspm_dir / 'Schaefer2018_400Parcels_17Networks_order_stcs.npy'

labels_stcs = np.load(file)


# load parcellation labels 

parc = 'Schaefer2018_400Parcels_17Networks_order' # cortical surface atlas we wish to use 

labels_parc = mne.read_labels_from_annot(
    subject=pnum_subj, 
    parc=parc, 
    subjects_dir=mri_subj_dir)

del labels_parc[401] # - removes medial wall
del labels_parc[0] # removes pointless header

# define threshold 
cutoff = .50 # USER INPUT #

thresh_val = (np.max(labels_stcs[:,900])) * cutoff 
interest_stcs = labels_stcs[:,750:960]
spike_window = len(interest_stcs[0,:])

is_spiking = interest_stcs < thresh_val # gets all valuess less than cutoff 


# find source and spread
spike_times = np.full(is_spiking.shape[0], np.NaN)

for parc_idx in range(is_spiking.shape[0]):

    parc_spiking = is_spiking[parc_idx,:]
    try:
        spike_contenders = np.argwhere(np.diff(parc_spiking)).squeeze() 
        last_index = spike_contenders.size
        spike_time = spike_contenders[last_index-1]
        spike_times[parc_idx] = spike_time
    except IndexError:
        continue
    
final_times = (spike_times - np.nanmin(spike_times))*(1000/600) 

np.savetxt(
    fname=(dspm_dir / 'propagation_delays_fs-order.csv'),
    X=final_times
)
