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


# ------ funcs ----------------------------------------------------------------
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

# mSSS ------------------------------------------------------------------------
def _combine_sss_basis(S_in1, S_in2):
    S_tot = [];
    thresh = 5e-7 #0.005 in Matlab, we will need to do some further investigating here once your code works
    U, s, Vh = np.linalg.svd(np.concatenate((S_in1,S_in2),axis=1))
    #apply threshold to limit dimensions of resulting basis
    for i in range(0, np.shape(s)[0]):
        ratio = s[i]/s[0]
        if ratio >= thresh:
            S_tot.append(U[:,i])
    return np.transpose(np.array(S_tot))


def mSSS_recon(raw, S_tot):
    phi_0 = raw.get_data(picks='meg')
    pS = np.linalg.pinv(S_tot)
    XN = pS @ phi_0
    data_mSSS = np.real(S_tot@XN)
    return data_mSSS

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

def fosters_inverse_msss(raw,origins):
    """
    Parameters
    ----------
    raw : mne.raw structure
        full raw meg file, ex. "fif", from recording with raw.info["bads"] indicated
    origins: list of two 1x3 np.array
        two origins optimized using "fit_sphere_mri.py" to fit the internal brain
        without encoraching on sensors
    
    Returns
    -------
    raw_fos : mne.raw structure
        raw strucutre with the MEG data updated with the Fosters Inverse 
        preprocessed data, raw.info structure updated to indicate some type of
        Maxwell Filtering/SSS preprocessing has occured. Channels marked "bad" 
        are dropped
    """
    ## calculate sensor noise covariance
    N = mne.compute_raw_covariance(raw,rank="info",method='empirical')["data"]
    ## create data strcutre, indicates in "info" that some preprocessing akin to SSS has happened
    raw_fos_msss = mne.preprocessing.maxwell_filter(raw, origin=(0.,0.,0.), int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize='in', ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)  # just to get the info to indicate some Maxwell filtering was done etc.
    #assert raw.info["bads"] == [] # double check bads were dropped
    ## extract raw data matrix from MEG channels
    phi_0 = raw_fos_msss.get_data(picks='meg')
    
    ## calculate mSSS matrix 
    [S_1, f, reg_moments, n_use_in1]=mne.preprocessing.compute_maxwell_basis(raw.info, origin=origins[0], int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
    [S_2, f, reg_moments, n_use_in2]=mne.preprocessing.compute_maxwell_basis(raw.info, origin=origins[1], int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
    S_tot= _combine_sss_basis(S_1[:, :n_use_in1], S_2[:, :n_use_in2])
    
    [S, f, reg_moments, n_use_in]=mne.preprocessing.compute_maxwell_basis(raw.info, origin=[0,0,0], int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize=None, ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)
    ## setup Foster's Inverse- calculate Matrix B and vector b
    # #for full basis
    S_out = S[:, n_use_in:]
    S=np.concat((S_tot,S_out),axis=1)
    # #for internal only
    # S=S_tot
    pS = np.linalg.pinv(S)
    XN = pS @ phi_0
    
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
    data_fosters = np.real(S_tot@x_bar[:np.size(S_tot,1),:])
    
    ## isolate MEG channels 
    meg_picks = mne.pick_types(raw.info, meg=True)
    ## put new Foster's inverse recon data into "raw" structure
    raw_fos_msss._data[meg_picks] = data_fosters
    
    ## cleanup
    del data_fosters
    
    return raw_fos_msss


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
    sample_files = ['sub-S027/sub-S027_file-Listening_raw.fif',
                    'sub-S027/sub-S027_file-ArmsRaisingAlternate_raw.fif',
                    'sub-S027/sub-S027_file-ArmsRaisingTogether_raw.fif',
                    'sub-S027/sub-S027_file-HandClenchAlternate_raw.fif',
                    'sub-S027/sub-S027_file-FootTapping_raw.fif',
                    'sub-S027/sub-S027_file-EyeBlink_raw.fif',
                    'sub-S027/sub-S027_file-JawMovement_raw.fif']
    
    
    ##-- define constants -----------------------------------------------
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
    baseline=(None,0)
    
    freq_min = None
    freq_max = 100
    
    ## -- Look at data + try some preprocessing to clean it -------------------
    for file in sample_files:
        raw = mne.io.read_raw_fif(os.path.join(sample_dir, file),preload=False, allow_maxshield='no')
        subject = file[4:8]
        task = file[23:]
        
        #find any NaN channels
        bads_list=[]
        for i in range(0,raw.info["nchan"]):
            ch_pos = raw.info["chs"][i]["loc"][:3]
            if np.isnan(ch_pos).any():
                bads_list.append(raw.info["chs"][i]["ch_name"])
        raw.drop_channels(bads_list)
        bads = raw.info["bads"]
        #drop bad channels
        raw.drop_channels(bads)

        
        ## high and low - pass, notch filter raw data--------------------------
        raw.load_data().filter(l_freq=freq_min, h_freq=None)
        raw.load_data().filter(l_freq=None, h_freq=freq_max)

        ## SSP ----------------------------------------------------------------
        raw_ssp = ssp_filter(raw)
        
        ## Traditional SSS ----------------------------------------------------
        raw_sss = sss_prepros(raw)     
        
        ## Foster's inverse with traditional SSS ------------------------------
        # This is my method, not published (yet)
        raw_fos = fosters_inverse(raw)
        
        ## joint plot compare -------------------------------------------------
        fig, axes = plt.subplots(4, 1, sharey=True, layout="constrained", figsize=(10, 10))
        for ax, data, title in zip(axes, [raw, raw_ssp, raw_sss,raw_fos], ["Raw Empty Room Data", "SSP Preprocessed","SSS Preprocessed", "Fosters Inverse Preprocessed"]):
            fig = data.compute_psd(fmax=100).plot(average=True, amplitude=False, picks="data", exclude="bads", axes=ax)
            ax.set(title=subject+', '+task+', '+title, xlim=(0, 100), ylim=(0,90))
            
            
            
            
            
            
            


