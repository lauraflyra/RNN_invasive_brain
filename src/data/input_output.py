import pandas as pd
import numpy as np
import torch

'''
Read processed time series of FFT features, movement traces and updrs. Structure such that FFT and updrs can be
given as input to RNN_SLP, and movement traces can be predicted. 
Take 1 min recording from multiple patients. 
Prepare leave one patient out cross validation. 
'''

PATH_UPDRS = "/home/lauraflyra/Documents/BCCN/Decoding_Parkinsons_UPDRS/RNN_invasive_brain/src/data/df_updrs.csv"
df_updrs = pd.read_csv(PATH_UPDRS, index_col=0)  # get the uprds scores for all patients

PATH_RMAP = "/home/lauraflyra/Documents/BCCN/Decoding_Parkinsons_UPDRS/RNN_invasive_brain/src/data/df_best_func_rmap_ch.csv"
df_rmap = pd.read_csv(PATH_RMAP, index_col=0)  # get the best channel from each patient in all 3 cohorts

df_all_channels = np.load(
    "/home/lauraflyra/Documents/BCCN/Decoding_Parkinsons_UPDRS/RNN_invasive_brain/src/data/channel_all.npy",
    allow_pickle=True)

# df_all_channelss[()]['Berlin']['001']['ECOG_L_1_2_SMC_AT-avgref']['sub-001_ses-EcogLfpMedOn01_task-SelfpacedForceWheel_acq-StimOff_run-01_ieeg']['feature_names']
# has the fft time series for all ecog channels for all subjects, in different runs
# calculated feature names -> 'feature_names', data -> 'data', movement -> 'label'

cohort = 'Pittsburgh'
sub_list = np.asarray(df_rmap[df_rmap['cohort'] == cohort]['sub'])
best_ch_list = list(df_rmap[df_rmap['cohort'] == cohort]['ch'])
df_cohort = df_all_channels[()][cohort]

data = []
updrs = []
behavior = []

for idx, sub in enumerate(sub_list):
    for key in df_cohort[sub][best_ch_list[idx]].keys():
        if "run-0_" in key:
            # take only the ones that have run-0 with more than 3 minutes, i.e shape > (1800,7)
            if df_cohort[sub][best_ch_list[idx]][key]['data'].shape[0] >= 2400:
                updrs.append(
                    df_updrs[(df_updrs["cohort"] == cohort) & (df_updrs["sub"] == sub)]["UPDRS_total"].tolist())
                # take the second and third minutes of the recordings
                data.append(df_cohort[sub][best_ch_list[idx]][key]['data'][600:2400, :])
                behavior.append(df_cohort[sub][best_ch_list[idx]][key]['label'][600:2400])

# input should be in shape (time_steps, n_batches, n_features)

updrs = torch.from_numpy(np.asarray(updrs).reshape(-1,1)).float()
data = np.asarray(data)
n_batches, time_steps, n_features = data.shape
neural = torch.from_numpy(data.reshape(time_steps, n_batches, n_features)).float()
behavior = torch.from_numpy(np.asarray(behavior, dtype=float).reshape(time_steps, n_batches, 1)).float()




