#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:13:58 2026

@author: alexandria


2026 Gwilliams Movement Pilot
"""


# --- Dependencies ------------------------------------------------------------
import os
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd


# ------ funcs ----------------------------------------------------------------
def get_events_fif(raw,file):
    ## TODO: go through evends, made pandas df
    event_code_list = events[:, 2]
    event_code_updates = np.zeros_like(event_code_list)
    special_codes = np.array([63,110])
    # ei = 0
    # while ei < len(event_code_list):
    #     event = event_code_list[ei]
    #     if event in special_codes:
    #         event_code_updates[ei+1:ei+5] = event
    #         event_code_updates[ei] = 201  # code for what was the condition label
    #         ei += 4  # skip next 4 positions 
    #     else:
    #         ei += 1  # just advance by 1 if no match
    # events[:, 2] = event_code_updates
    # # get just the code part
    # trigger_codes = events[:, 2]
    # blocks = trigger_codes.reshape(-1, 5)
    # blocks_rearranged = blocks[:, [1, 2, 3, 4, 0]]
    # result = blocks_rearranged.flatten()
    # events[:, 2] = result
    ## make event codes interpretable
    code_dict = {63: "code1",
                 110: "code2",
                 255: "code3"}
    
    # # make into a nice pandas dataframe
    # events_df = pd.DataFrame()
    # events_df['code'] = events[:, 2]
    # events_df['condition'] = [code_dict[c].split('_')[0] for c in events[:, 2]]
    # events_df['font'] = [code_dict[c].split('_')[1] for c in events[:, 2]]
    
    return

def highpass_filter_opm(raw):
    freq_min = 1
    freq_max = 70
    raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    return raw
    
def ssp_filter(raw):
    # SSP projector
    proj = mne.compute_proj_raw(raw, start=0, stop=None, duration=1, n_grad=0, n_mag=1, n_eeg=0, reject=None, flat=None, n_jobs=None, meg='separate', verbose=None)
    raw_proj = raw.copy().add_proj(proj)
    return raw_proj

def sss_prepros(raw):
    raw_sss = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.), int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
    return raw_sss

## Fosters --------------------------------------------------------------------
def _do_inverse(raw,N):
    """
    Parameters
    ----------
    raw : mne.raw structure
        full raw meg file, ex. "fif", from recording with raw.info["bads"] indicated
    N : 2D square matrix, (number of sensors) X (number of sensors)
        Sensor noise covariance matrix, calculated using empircial covariance
        implemented in mne.compute_raw_covariance

    Returns
    -------
    data_fosters : 2D matrix, (number of sensors) X (time)
        Matrix containing data corresponding to each MEG channel over time after
        reconstruction with Fosters Inverse preprocessing
    """
    ## extract raw data matrix from MEG channels
    phi_0 = raw.get_data(picks='meg')
    ## calculate SSS matrix S and multiple moments with reccomended params
    [S, pS, reg_moments, n_use_in]=mne.preprocessing.compute_maxwell_basis(raw.info, origin=(0.,0.,0.), int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
    
    ## setup Foster's Inverse- calculate Matrix B and vector b
    S = S[:, :n_use_in]
    XN = pS[:n_use_in,:] @ phi_0
    ## for full S
    # XN = pS @ phi_0
    alpha = np.transpose(XN)
    alpha_cov_norm = np.cov(XN)
    S_star = np.transpose(np.conj(S))
    first = np.linalg.pinv(S@alpha_cov_norm@S_star +N)
    B = alpha_cov_norm @ S_star @ first
    m_alpha = np.transpose(np.mean(alpha,0))
    b = m_alpha - B@S@m_alpha
    x_bar = np.zeros_like(XN)
    
    ## calculate Foster's Inverse estimate of multipole moments
    for i in range(0,np.shape(phi_0)[1]):
        x_bar[:,i]=B@phi_0[:,i] + b
    
    ## use new estimate to reconstruct internal data
    data_fosters = np.real(S[:, :n_use_in]@x_bar[:n_use_in,:])
    return data_fosters
    
def fosters_inverse(raw):
    """
    Parameters
    ----------
    raw : mne.raw structure
        full raw meg file, ex. "fif", from recording with raw.info["bads"] indicated
    
    Returns
    -------
    raw_fos : mne.raw structure
        raw strucutre with the MEG data updated with the Fosters Inverse 
        preprocessed data, raw.info structure updated to indicate some type of
        Maxwell Filtering/SSS preprocessing has occured. Channels marked "bad" 
        are dropped
    """
    ## calculate sensor noise covariance
    N = mne.compute_raw_covariance(raw,tmin=0, tmax=10,rank="info",method='empirical')["data"]
    ## create data strcutre, indicates in "info" that some preprocessing akin to SSS has happened
    raw_fos = mne.preprocessing.maxwell_filter(raw, origin=(0.,0.,0.), int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)  # just to get the info to indicate some Maxwell filtering was done etc.
    #assert raw.info["bads"] == [] # double check bads were dropped
    
    ## Do foster's inverse!
    foster_sss_data= _do_inverse(raw, N)
    
    ## isolate MEG channels 
    meg_picks = mne.pick_types(raw.info, meg=True)
    ## put new Foster's inverse recon data into "raw" structure
    raw_fos._data[meg_picks] = foster_sss_data
    
    ## cleanup
    del foster_sss_data
    
    return raw_fos

#### calculate SNR
def calculate_snr(signal, noise):
    signal_power = np.mean(np.var(signal))
    noise_power = np.mean(np.var(noise))
    snr = signal_power / noise_power
    return snr

    
    
#########################
# --- MAIN ------------------------------------------------------------
if __name__ == '__main__':
    ## define raw file directory and raw file names
    sample_dir = '/Users/alexandria/Documents/STANFORD/DATA/2026_Gwilliams_MovementPilot/NoTask'
    subjects = ['sub-S027','sub-S026']
    sample_files = ['sub-S027/sub-S027_file-Listening_raw.fif',
                    'sub-S027/sub-S027_file-ArmsRaisingAlternate_raw.fif',
                    'sub-S027/sub-S027_file-ArmsRaisingTogether_raw.fif',
                    'sub-S027/sub-S027_file-HandClenchAlternate_raw.fif',
                    'sub-S027/sub-S027_file-FootTapping_raw.fif',
                    'sub-S027/sub-S027_file-EyeBlink_raw.fif',
                    'sub-S027/sub-S027_file-JawMovement_raw.fif']

    
    ##-- define constants -----------------------------------------------
    compare_prepross=True
    
    n_trials = len(sample_files)
    trigger_chan = 'di2'  
    tmin = -0.1
    tmax = 0.5
    
    ##-- specify plotting args -----------------------------------------------
    ts_args = ts_args = dict(
        time_unit="s",
        #ylim=dict(mag=(-400, 400)),
        gfp=True
    )
    topomap_args = dict(
        time_unit="s",
        vlim=(-500,500)
    )
    baseline=None
    
    freq_min = 0.5
    freq_max = 80
    
    ## -- Look at data + try some preprocessing to clean it -------------------
    raws =[]
    tasks =[]
    for file in sample_files:
        raw = mne.io.read_raw_fif(os.path.join(sample_dir, file),preload=False, allow_maxshield='no')
        subject = file[4:8]
        task = file[23:]
        tasks.append(task)
        
        #find any NaN channels
        bads_list=[]
        for i in range(0,raw.info["nchan"]):
            ch_pos = raw.info["chs"][i]["loc"][:3]
            if np.isnan(ch_pos).any():
                bads_list.append(raw.info["chs"][i]["ch_name"])
        raw.drop_channels(bads_list)
        bads = raw.info["bads"]
        raw.drop_channels(bads)

        ## high and low - pass, notch filter raw data--------------------------
        raw.load_data().filter(l_freq=freq_min, h_freq=None)
        raw.load_data().filter(l_freq=None, h_freq=freq_max)
        meg_picks = mne.pick_types(raw.info, meg=True)
        raw.notch_filter(freqs=60, picks=meg_picks)
        raws.append(raw)
        fig = raw.compute_psd(fmax=freq_max).plot(average=False, amplitude=False, picks="data", exclude="bads")

        ## -- get events ---------------------------------------------------
        ## TODO: parse events
        ## [events_df,events,task] = get_events_fif(raw,file)
        events = mne.find_events(raw, stim_channel=trigger_chan, shortest_event=1)
        info = raw.info
        picks = 'mag'
        reject_criteria = dict(mag=4000e-15)
        
        # --- 3. Make Epochs and Evokeds --------------------------------------
        ## very basic example -- we will do something fancier
        tmin = -0.05  # start of each epoch (200ms before the trigger)
        tmax = 0.3  # end of each epoch (600ms after the trigger)
        baseline = None
        # can add rejection criteria based on P-to-P signal        
        # separate out by event ID
        epochs = mne.Epochs(raw, events, 
                    tmin=tmin, tmax=tmax,
                    baseline=baseline,
                    reject=None,
                    preload=True)
        evoked = epochs.average()
        ## specify plotting args
        ts_args = ts_args = dict(time_unit="s") 
        topomap_args = dict(time_unit="s") 
        fig = evoked.plot_joint(times="peaks", ts_args=ts_args, topomap_args=topomap_args, title= subject+ ' Task: '+ task)
        
     
        
        if compare_prepross:
            ## SSP ----------------------------------------------------------------
            raw_ssp = ssp_filter(raw)
            
            ## Traditional SSS ----------------------------------------------------
            raw_sss = sss_prepros(raw)     
            
            ## Foster's inverse with traditional SSS ------------------------------
            # This is my method, not published (yet)
            raw_fos = fosters_inverse(raw)
            
            ## joint plot compare -------------------------------------------------            
            # fig, axes = plt.subplots(4, 1, sharey=True, layout="constrained", figsize=(10, 10))
            # for ax, data, title in zip(axes, [raw, raw_ssp, raw_sss,raw_fos], ["Raw Empty Room Data", "SSP Preprocessed","SSS Preprocessed", "Fosters Inverse Preprocessed"]):
            #     fig = data.compute_psd(fmax=freq_max).plot(average=True, amplitude=False, picks="data", exclude="bads", axes=ax)
            #     ax.set(title=subject+', '+task+', '+title, xlim=(0, freq_max), ylim=(0,90))
            for raw,method in zip([raw_ssp,raw_sss,raw_fos],["SSP","SSS","Fosters"]):
                epochs = mne.Epochs(raw, events, 
                            tmin=tmin, tmax=tmax,
                            baseline=baseline,
                            reject=None,
                            preload=True)
                evoked = epochs.average()
                ## specify plotting args
                ts_args = ts_args = dict(time_unit="s") 
                topomap_args = dict(time_unit="s") 
                fig = evoked.plot_joint(times="peaks", ts_args=ts_args, topomap_args=topomap_args, title= subject+ ' '+ method +' Task: '+ task)
                

## just look at raws for each movement            
fig, axes = plt.subplots(n_trials, 1, sharey=True, layout="constrained", figsize=(10, 20))
for ax, raw, task in zip(axes, raws, tasks):
    fig = raw.compute_psd(fmax=freq_max).plot(average=False, amplitude=False, picks="data", exclude="bads", axes=ax)
    ax.set(title=subject+', '+task, xlim=(0, freq_max), ylim=(0,90))
               
            
            
            
            


