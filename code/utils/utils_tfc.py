import os
import sys
import re
import shutil
import warnings
from copy import deepcopy
from os import listdir
from os.path import isdir, isfile
from os.path import join as pjoin
from pathlib import Path
from typing import Callable, List, Optional, Union
from uuid import uuid4
import dask.array as darr
import warnings
from joblib import Parallel, delayed
import time
from pathlib import PureWindowsPath, PurePosixPath
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xarray as xr
import pandas as pd

from scipy.interpolate import CubicSpline
from scipy.stats import binned_statistic_2d
import statsmodels.api as sm
from scipy.stats import skew, pearsonr
from scipy.signal import find_peaks

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, GridSearchCV
from scipy.ndimage import binary_dilation
from scipy.stats import median_abs_deviation
from scipy.interpolate import interp1d
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
from scipy.interpolate import pchip_interpolate
from scipy.ndimage import label, gaussian_filter1d, median_filter
from scipy.signal import savgol_filter


# Variable definition
#Define TFC protocols and process Behavior camera parameters
TFC_proto = {
    'du_baseline': 240 *1000,
    'du_cue': 20 *1000,
    'du_trace': 20 *1000,
    'du_shock': 3 *1000,
    'du_iti_batch1': (60-3) *1000,
    'du_iti_batch2': (120-3) *1000,
    'trials': 6,
    'trials_resp': 6, # How many trials used for checking responsive cells
    'trials_disp': 6, # With processed responvise cell, display their activity in how many trials
    'trials_start': 0, # 0: start from the first trial; 1: from 2nd tiral; 2: from 3rd trial...
    'du_iti_1': 20 *1000,
    'du_iti_2': 20 *1000,
    'du_iti_3': 17 *1000, 
    'du_re_baseline1': 120*1000, # For recall session, batch 1
    'du_re_cue1': 60*1000,    
    'du_re_baseline2': 180*1000, # For recall session, batch 2
    'du_re_cue2': 30*1000,    
    'du_re_iti': 120*1000,
    're_trials': 4,
    're_trials_resp': 4 #  1 for 1st trial with baseline; 4 for all 4 tirals with baseline (start to end)
}

TFC_proto_60 = {
    'du_baseline': 240 *1000,
    'du_cue': 20 *1000,
    'du_trace': 60 *1000,
    'du_shock': 3 *1000,
    'du_iti_batch1': 0, # No batch1
    'du_iti_batch2': (120-3) *1000,
    'trials': 6,
    'trials_resp': 6, # How many trials used for checking responsive cells
    'trials_disp': 6, # With processed responvise cell, display their activity in how many trials
    'trials_start': 0, # 0: start from the first trial; 1: from 2nd tiral; 2: from 3rd trial...
    'du_iti_1': 20 *1000,
    'du_iti_2': 20 *1000,
    'du_iti_3': 17 *1000, 
    'du_re_baseline1': 120*1000, # For recall session, batch 1
    'du_re_cue1': 60*1000,    
    'du_re_baseline2': 180*1000, # For recall session, batch 2
    'du_re_cue2': 30*1000,    
    'du_re_iti': 120*1000,
    're_trials': 4,
    're_trials_resp': 4 #  1 for 1st trial with baseline; 4 for all 4 tirals with baseline (start to end)
}

TFC_proto_5 = {
    'du_baseline': 240 *1000,
    'du_cue': 20 *1000,
    'du_trace': 5 *1000,
    'du_shock': 3 *1000,
    'du_iti_batch1': 0, # No batch1
    'du_iti_batch2': (120+2) *1000,
    'trials': 6,
    'trials_resp': 6, # How many trials used for checking responsive cells
    'trials_disp': 6, # With processed responvise cell, display their activity in how many trials
    'trials_start': 0, # 0: start from the first trial; 1: from 2nd tiral; 2: from 3rd trial...
    'du_iti_1': 20 *1000,
    'du_iti_2': 20 *1000,
    'du_iti_3': 17 *1000, 
    'du_re_baseline1': 120*1000, # For recall session, batch 1
    'du_re_cue1': 60*1000,    
    'du_re_baseline2': 180*1000, # For recall session, batch 2
    'du_re_cue2': 30*1000,    
    'du_re_iti': 120*1000,
    're_trials': 4,
    're_trials_resp': 4 #  1 for 1st trial with baseline; 4 for all 4 tirals with baseline (start to end)
}

TFC_proto_0 = {
    'du_baseline': 240 *1000,
    'du_cue': 20 *1000,
    'du_trace': 0 *1000,
    'du_shock': 3 *1000,
    'du_iti_batch1': 0, # No batch1
    'du_iti_batch2': (120-3) *1000,
    'trials': 6,
    'trials_resp': 6, # How many trials used for checking responsive cells
    'trials_disp': 6, # With processed responvise cell, display their activity in how many trials
    'trials_start': 0, # 0: start from the first trial; 1: from 2nd tiral; 2: from 3rd trial...
    'du_iti_1': 20 *1000,
    'du_iti_2': 20 *1000,
    'du_iti_3': 17 *1000, 
    'du_re_baseline1': 120*1000, # For recall session, batch 1
    'du_re_cue1': 60*1000,    
    'du_re_baseline2': 180*1000, # For recall session, batch 2
    'du_re_cue2': 30*1000,    
    'du_re_iti': 120*1000,
    're_trials': 4,
    're_trials_resp': 4 #  1 for 1st trial with baseline; 4 for all 4 tirals with baseline (start to end)
}


def open_minian(
    dpath: str, post_process: Optional[Callable] = None, return_dict=False
) -> Union[dict, xr.Dataset]:
    """
    Load an existing minian dataset.
    Returns
    -------
    ds : Union[dict, xr.Dataset]
        The resulting dataset. If `return_dict` is `True` it will be a `dict`,
        otherwise a `xr.Dataset`.

    See Also
    -------
    xarray.open_zarr : for how each directory will be loaded as `xr.DataArray`
    xarray.merge : for how the `xr.DataArray` will be merged as `xr.Dataset`
    """
    if isfile(dpath):
        ds = xr.open_dataset(dpath, consolidated=False).chunk()
    elif isdir(dpath):
        dslist = []
        for d in listdir(dpath):
            arr_path = pjoin(dpath, d)
            if isdir(arr_path):
                arr = list(xr.open_zarr(arr_path, consolidated=False).values())[0]
                arr.data = darr.from_zarr(
                    os.path.join(arr_path, arr.name), inline_array=True
                )
                dslist.append(arr)
        if return_dict:
            ds = {d.name: d for d in dslist}
        else:
            ds = xr.merge(dslist, compat="no_conflicts")
    if (not return_dict) and post_process:
        ds = post_process(ds, dpath)
    return ds

def xrconcat_recursive(var: Union[dict, list], dims: List[str]) -> xr.Dataset:
    """
    Recursively concatenate `xr.DataArray` over multiple dimensions.

    Parameters
    ----------
    var : Union[dict, list]
        Either a `dict` or a `list` of `xr.DataArray` to be concatenated. If a
        `dict` then keys should be `tuple`, with length same as the length of
        `dims` and values corresponding to the coordinates that uniquely
        identify each `xr.DataArray`. If a `list` then each `xr.DataArray`
        should contain valid coordinates for each dimensions specified in
        `dims`.
    dims : List[str]
        Dimensions to be concatenated over.

    Returns
    -------
    ds : xr.Dataset
        The concatenated dataset.

    Raises
    ------
    NotImplementedError
        if input `var` is neither a `dict` nor a `list`
    """
    if len(dims) > 1:
        if type(var) is dict:
            var_dict = var
        elif type(var) is list:
            var_dict = {tuple([(v[d]).item() for d in dims]): v for v in var} #np.asscalar
        else:
            raise NotImplementedError("type {} not supported".format(type(var)))
        try:
            var_dict = {k: v.to_dataset() for k, v in var_dict.items()}
        except AttributeError:
            pass
        data = np.empty(len(var_dict), dtype=object)
        for iv, ds in enumerate(var_dict.values()):
            data[iv] = ds
        index = pd.MultiIndex.from_tuples(list(var_dict.keys()), names=dims)
        var_ps = pd.Series(data=data, index=index)
        xr_ls = []
        for idx, v in var_ps.groupby(level=dims[0]):
            v.index = v.index.droplevel(dims[0])
            xarr = xrconcat_recursive(v.to_dict(), dims[1:])
            xr_ls.append(xarr)
        return xr.concat(xr_ls, dim=dims[0])
    else:
        if type(var) is dict:
            var = list(var.values())
        return xr.concat(var, dim=dims[0])

def position_cutoff(df, scorer, bodyparts, cutoff_thre=0.7, scale_f=0.23, fps=15, max_speed_cm_s=150):
    """
    Cleans tracking data by applying a likelihood threshold followed by a 
    kinematic speed check to remove impossible physical movements.
    Parameters:
    - df: The DeepLabCut pandas DataFrame.
    - scorer: The name of the scorer (string).
    - bodyparts: List of body parts to process (e.g., ['nose', 'neck', 'tailbase']).
    - cutoff_thre: Minimum likelihood score to keep a data point.
    - max_speed_cm_s: Maximum biologically plausible speed in cm/s.
    - fps: Frames per second of the recorded video.
    - scale_f: Conversion factor from pixels to centimeters.
  
    Returns:
    - df: The cleaned DataFrame with invalid positions replaced by np.nan.
    """   
    for bp in bodyparts:
        # ==========================================
        # Step 1: Likelihood Cutoff
        # ==========================================
        low_conf_mask = df[scorer][bp]['likelihood'] < cutoff_thre
        df.loc[low_conf_mask, (scorer, bp, 'x')] = np.nan
        df.loc[low_conf_mask, (scorer, bp, 'y')] = np.nan        
        # ==========================================
        # Step 2: Kinematic Speed Check
        # ==========================================
        # Calculate frame-to-frame differences. 
        # (Note: diff() against an np.nan results in np.nan)
        dx = df[(scorer, bp, 'x')].diff()
        dy = df[(scorer, bp, 'y')].diff()        
        # Calculate Euclidean distance in pixels
        dist_px = np.sqrt(dx**2 + dy**2)        
        # Convert distance to speed (cm per second)
        speed_cm_s = (dist_px * scale_f) * fps        
        # Create mask for impossible speeds. 
        # (np.nan > max_speed_cm_s safely evaluates to False)
        impossible_mask = (speed_cm_s > max_speed_cm_s).values     
        # Set speed outliers to NaN
        df.loc[impossible_mask, (scorer, bp, 'x')] = np.nan
        df.loc[impossible_mask, (scorer, bp, 'y')] = np.nan        
    return df
    
def hampel_filter(data, window_size=3, n_sigma=3):
    """ Hampel filter using numpy masked arrays. """
    masked_data = np.ma.masked_invalid(data)
    filtered_data = data.copy()
    k = 1.4826   
    for i in range(len(data)):
        if np.ma.is_masked(masked_data[i]):
            continue            
        start_index = max(0, i - window_size)
        end_index = min(len(data), i + window_size + 1)        
        window = masked_data[start_index:end_index]
        
        median_val = np.ma.median(window)
        mad = np.ma.median(np.abs(window - median_val))       
        if mad == 0:
            continue           
        if np.abs(data[i] - median_val) > n_sigma * k * mad:
            filtered_data[i] = median_val            
    return filtered_data

def pchip_interpolation(data):
    """ 
    PCHIP interpolation prevents the wild overshoots common with Cubic Splines 
    during long occlusions.
    """
    x = np.arange(data.size)
    valid_mask = ~np.isnan(data)   
    # If there is no valid data or too little to interpolate, return as is
    if valid_mask.sum() < 2:
        return data        
    x_valid = x[valid_mask]
    y_valid = data[valid_mask]
    
    # Interpolate for the missing points
    y_filled = data.copy()
    missing_mask = np.isnan(data)
    
    # Only interpolate if there are actually missing values
    if missing_mask.any():
        # Extrapolate=False leaves trailing/leading NaNs as NaN rather than guessing
        y_filled[missing_mask] = pchip_interpolate(x_valid, y_valid, x[missing_mask])
        # Fill the Leading and Trailing NaNs
        y_series = pd.Series(y_filled)
        y_filled = y_series.bfill().ffill().values
        
    return y_filled

def calc_speed_gradient(t, x, y=None):
    """
    Calculates speed using central difference (np.gradient).
    This perfectly aligns speed with the timestamp index.
    """
    if y is None:
        y = np.zeros(t.size)
    # Calculate velocity components (dx/dt and dy/dt)
    v_x = np.gradient(x, t)
    v_y = np.gradient(y, t)
    
    # Calculate magnitude of the velocity vector
    speed = np.sqrt(v_x**2 + v_y**2)
    return speed 

def calc_position_idx(x, y, bin_num_x, bin_num_y, chamber_x_min, chamber_x_max, chamber_y_min, chamber_y_max, outlier_f=2.0):
    """
    Calculates a stable 1D spatial bin index (1 to bin_num_x*bin_num_y) for 2D coordinates.
    Guarantees fixed spatial geometry regardless of animal occupancy.
    """
    # 1. Define physical edges including the outlier buffer
    edges_x = np.linspace(chamber_x_min, chamber_x_max, num= bin_num_x+1)
    edges_y = np.linspace(chamber_y_min, chamber_y_max, num= bin_num_y+1)
    #Enlarge the boundary to include the possible outeliers
    edges_x[0] -= outlier_f
    edges_x[bin_num_x] += outlier_f
    edges_y[0] -= outlier_f
    edges_y[bin_num_y] += outlier_f

    # 2. Get 0-indexed column and row assignments using np.digitize
    # np.digitize returns 1 to bins. Subtract 1 to make it 0-indexed.
    idx_x = np.digitize(x, edges_x) - 1
    idx_y = np.digitize(y, edges_y) - 1

    # 3. Handle Extreme Outliers (DLC teleports)
    # If a tracking point jumps 50cm away, clip forces it into the nearest valid edge bin
    idx_x = np.clip(idx_x, 0, bin_num_x - 1)
    idx_y = np.clip(idx_y, 0, bin_num_y - 1)

    # 4. Algebraic flattening to 1D index
    # Formula: (Row_Index * Total_Columns) + Col_Index + 1
    # Chamber: Top-Left (0,0) -> 1. Bottom-Right (3,4) -> 20.
    bin_1d = (idx_y * bin_num_x) + idx_x + 1
        
    return bin_1d.astype(int)  
 
def map_ts(ts: pd.DataFrame) -> pd.DataFrame:
    """map frames from Cam1 to Cam0 with nearest neighbour using the timestamp file from miniscope recordings.
    
    Cam0--miniscope recording
    Cam1--Behav camera
    
    Parameters
    ----------
    ts : pd.DataFrame
        input timestamp dataframe. should contain field 'frameNum', 'camNum' and 'sysClock'
    
    Returns
    -------
    pd.DataFrame
        output dataframe. should contain field 'fmCam0' and 'fmCam1'
    """
    ts_sort = ts.sort_values("sysClock")
    ts_sort["ts_behav"] = np.where(ts_sort["camNum"] == 1, ts_sort["sysClock"], np.nan)
    ts_sort["ts_forward"] = ts_sort["ts_behav"].ffill()
    ts_sort["ts_backward"] = ts_sort["ts_behav"].bfill()

    ts_sort["diff_forward"] = np.absolute(ts_sort["sysClock"] - ts_sort["ts_forward"])
    ts_sort["diff_backward"] = np.absolute(ts_sort["sysClock"] - ts_sort["ts_backward"])
    ts_sort["fm_behav"] = np.where(ts_sort["camNum"] == 1, ts_sort["frameNum"], np.nan)
    ts_sort["fm_forward"] = ts_sort["fm_behav"].ffill()
    ts_sort["fm_backward"] = ts_sort["fm_behav"].bfill().ffill() # .ffill() again when the last value is NaN

    ts_sort["fmCam1"] = np.where(
        ts_sort["diff_forward"] < ts_sort["diff_backward"],
        ts_sort["fm_forward"],
        ts_sort["fm_backward"])
    ts_map = (
        ts_sort[ts_sort["camNum"] == 0][["frameNum", "fmCam1"]]
        .dropna()
        .rename(columns=dict(frameNum="fmCam0"))
        .astype(dict(fmCam1=int)))
    #ts_map["fmCam0"] = ts_map["fmCam0"]# - 1
    #ts_map["fmCam1"] = ts_map["fmCam1"]# - 1
    return ts_map

def process_animial_behav(dpath_beh: str, output_path, scale_f = 0.055, fps=15, bin_num_x = 5, bin_num_y = 4, DLCscorer='DLC_Resnet50_TFCAug13shuffle1_snapshot_best-180'):
    # Get the behavior data inlcuding (head coordinates, speed, position bin)
    print(dpath_beh, '-----processing')
    try:
        df_beh = pd.read_hdf(os.path.join(dpath_beh, 'behavConcat' + DLCscorer + '.h5'))
        df_ts = pd.read_csv(os.path.join(dpath_beh, "timeStamps.csv")) 
    except:
        print("file missing under {}".format(dpath_beh))
        return False

    scorer = df_beh.columns.get_level_values(0)[0]
    bodyparts = ['leftear', 'rightear', 'tailbase']
    # 01 cutoff the unqualified data
    cutoff_thre = 0.7
    df_beh = position_cutoff(df_beh, scorer, bodyparts, cutoff_thre, scale_f, fps, max_speed_cm_s=150)
    for bp in bodyparts:
        print(f"Processing: {bp}")
        for axis in ['x', 'y']:        
            data = df_beh[(scorer, bp, axis)].values # Extract as numpy array        
            # Step A: Remove outliers
            data = hampel_filter(data, window_size=3) # Half-window of 3 (7 frames total)        
            # Step B: Interpolate missing data (PCHIP prevents overshoots) and also fill the Leading and Trailing NaNs 
            data = pchip_interpolation(data)        
            # Step C: Smooth the continuous data
            data = gaussian_filter1d(data, sigma=2)        
            # Step D: Scale and assign back to dataframe
            # PCHIP can shoot to infinity if missing data exists at the start/end of the video.
            # scale the data, then force it inside a physical 100cm box.
            scaled_data = data * scale_f
            if axis =='x':
                scaled_data = np.clip(scaled_data, 0.0, 35.0) 
            if axis =='y':
                scaled_data = np.clip(scaled_data, 0.0, 28.0)
            # Step D: Scale and assign back to dataframe
            df_beh.loc[:, (scorer, bp, axis)] = scaled_data## chamber length 30cm/ 500 pixels (around, usually 514 pixels)
    # --- 03. Calculate location, speed, and head direction ---
    behav = pd.DataFrame()
    behav['ts'] = df_ts['Time Stamp (ms)'].values / 1000.0 # unit (second)
    # 1. Body Center Location
    behav['X'] = (df_beh[scorer]['leftear']['x'].values + df_beh[scorer]['rightear']['x'].values) / 2.0
    behav['Y'] = (df_beh[scorer]['leftear']['y'].values + df_beh[scorer]['rightear']['y'].values) / 2.0
    # 2. Speed Calculation & Smoothing cm/s
    behav['speed'] = calc_speed_gradient(behav['ts'].values, behav['X'].values, behav['Y'].values)
    # Use a slightly heavier sigma for speed due to derivative noise amplification
    behav['speed'] = gaussian_filter1d(behav['speed'].values, sigma=3)        

    # 3. Calculate Body position and speed
    behav['X_body'] = (behav['X'].values + df_beh[scorer]['tailbase']['x'].values) / 2.0
    behav['Y_body'] = (behav['Y'].values + df_beh[scorer]['tailbase']['y'].values) / 2.0
    behav['speed_body'] = calc_speed_gradient(behav['ts'].values, behav['X_body'].values, behav['Y_body'].values)
    behav['speed_body'] = gaussian_filter1d(behav['speed_body'].values, sigma=3) 
    
    # A mouse generally cannot exceed 100 cm/s in a small chamber. 
    # This prevents single-frame tracking artifacts from breaking the Y-axis of the plots.
    behav['speed'] = np.clip(behav['speed'].values, 0.0, 150.0)
    behav['speed_body'] = np.clip(behav['speed_body'].values, 0.0, 150.0)
    
    # 4. Get the animal position in bin index
    #bin_num_x = 5 #conditioning chamber 30cm*24cm, bin width 6cm, bin_num_y = 4
    #The positions of the chamber edge. x: (60, 600); y: (40, 480) based on the recording videos
    chamber_x_min = 60 * scale_f
    chamber_x_max = 600 * scale_f
    chamber_y_min = 40 * scale_f
    chamber_y_max = 480 * scale_f
    outlier_f = 2 # 2cm for a buffer to include the outliers to the current bins
    behav['posi'] = calc_position_idx(behav['X'], behav['Y'], bin_num_x, bin_num_y, chamber_x_min, chamber_x_max, chamber_y_min, chamber_y_max, outlier_f = outlier_f)
    
    animal = PureWindowsPath(dpath_beh).parts[-3]
    session = PureWindowsPath(dpath_beh).parts[-2]
    plot_behavior_diagnostics(behav, output_path, f'Behavior_preprocessing_overview_{animal}_{session}')
    
    behav.to_csv(os.path.join(dpath_beh, 'Processed_behavioral_data.csv'), index=False)
    print('Finished.')
    return True

def plot_behavior_diagnostics(behav, output_path, title):
    """
    Plots the tracked coordinates and resulting speed curves for QA checking.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # Plot 1: 2D Spatial Scatter of Body Parts
    ax_pos = axes[0]
    
    # plot a subset of frames (e.g., every 3rd frame) to prevent overplotting
    skip = 3
    ax_pos.scatter(behav['X'][::skip], behav['Y'][::skip], 
                   alpha=0.3, s=10, label='Head', color='#4091cf', edgecolors='none')
    ax_pos.scatter(behav['X_body'][::skip], behav['Y_body'][::skip], 
                   alpha=0.3, s=10, label='Body center', color='#8cba54', edgecolors='none')
    
    
    ax_pos.set_title('Denoised Body Part Positions', fontweight='bold')
    ax_pos.set_xlabel('X Position (cm)', fontweight='bold')
    ax_pos.set_ylabel('Y Position (cm)', fontweight='bold')
    
    # Camera coordinates usually have Y=0 at the top of the image
    ax_pos.invert_yaxis() 
    
    ax_pos.legend(frameon=True, loc='upper right')
    sns.despine(ax=ax_pos)
    
    # Plot 2: Speed Curves
    ax_spd = axes[1]
    ax_spd.plot(behav['ts'], behav['speed'], color='#4091cf', alpha=0.8, linewidth=1.5, label='Head Speed')
    ax_spd.plot(behav['ts'], behav['speed_body'], color='#e1703c', alpha=0.8, linewidth=1.5, label='Body Speed')
    
    ax_spd.set_title('Animal Speed Dynamics', fontweight='bold')
    ax_spd.set_xlabel('Time (s)', fontweight='bold')
    ax_spd.set_ylabel('Speed (cm/s)', fontweight='bold')
    ax_spd.set_xlim(0, behav['ts'].iloc[-1])
    ax_spd.legend(frameon=False, loc='upper right')
    sns.despine(ax=ax_spd)
    
    plt.tight_layout()
    #os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f'{title}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
def filter_abnormal_cells(sig_raw, sig_z_score, thre_skew=1.5, thre_plateau=15, min_peaks=3, fps=20):
    """
    Filters abnormal cells (Gaussian noise, plateaus, interneurons, silent cells) concisely.
    
    Parameters:
    -----------
    raw_traces : np.array (n_cells, n_frames)
    fps : float
    
    Returns:
    --------
    good_indices : np.array of indices to keep
    """
    # 1. check skewness
    n_cells = sig_raw.shape[0]
    n_frames = sig_raw.shape[1]
    is_skewed = skew(sig_raw, axis =1) >= thre_skew # Should be skewed to right

    # 2. Hyperactivity/Interneuron Filter (Vectorized)
    medians = np.median(sig_raw, axis=1, keepdims=True)
    stds = np.std(sig_raw, axis=1, keepdims=True)
    # Boolean mask of activity (n_cells, n_frames)
    is_active_mask = sig_raw > (medians + (3 * stds))

    # Check if active for > 20% of total time
    active_fraction = np.sum(is_active_mask, axis=1) / n_frames
    is_sparse = active_fraction < 0.20
    
    # 3. Plateau Artifact Filter and silent cells filter (Loop required for run-length encoding, but optimized)
    # Rejects cells with any single continuous event > 15 seconds
    max_duration_bins = int(thre_plateau * fps)
    no_plateaus = np.ones(n_cells, dtype=bool)
    no_silent = np.ones(n_cells, dtype=bool)
    for i in range(n_cells):
        if not (is_skewed[i] and is_sparse[i]): 
            continue # Skip calculation if already bad

        # Fast run-length finding
        active_trace = is_active_mask[i].astype(int)
        diffs = np.diff(np.hstack(([0], active_trace, [0])))
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]

        if len(run_starts) > 0:
            longest_event = np.max(run_ends - run_starts)
            if longest_event > max_duration_bins:
                no_plateaus[i] = False
                
        peaks, _ = find_peaks(sig_z_score[i,:], height=3, distance=10) # Z score >3 as real spikes
        if len(peaks) < min_peaks:
            no_silent[i] = False
    # Combine all filters
    good_mask = is_skewed & is_sparse & no_plateaus & no_silent
    good_indices = np.where(good_mask)[0]
    return good_indices #, is_skewed, is_sparse, no_plateaus, no_silent

def extrace_task_modulated_cells(Sig_raw, sig_z_score, TFC_proto, ts_trials, trial_dur, post_shock=20*1000, peak_thresh=3.0, min_active_trials=2, modu_index_thresh=0.15):
    #Pass 1: Cell must have peaks in >= 2 trials (Reliability).
    #Pass 2: Cell must have Modulation Index > 0.15 (Selectivity).
    
    n_cells = sig_z_score.unit_id.size
    trial_dur_task = TFC_proto['du_cue'] + TFC_proto['du_trace'] + TFC_proto['du_shock'] + post_shock # post-shock use 20s
    
    num_peaks = np.zeros(n_cells)
    rate_baseline = Sig_raw.where((ts_trials[0] < Sig_raw["ts"]) & (Sig_raw["ts"] < (ts_trials[0] + TFC_proto['du_baseline'])), drop=True).mean(dim='frame')
    rate_task = []
    for i in range(TFC_proto['trials_resp']):      
        start = ts_trials[0] + TFC_proto['du_baseline'] + i* trial_dur
        end = ts_trials[0] + TFC_proto['du_baseline'] + i* trial_dur + trial_dur_task
        
        sig_trial = sig_z_score.where((start < sig_z_score["ts"]) & (sig_z_score["ts"] < end), drop=True).compute()
        num_peaks += (sig_trial.max(dim='frame').values > peak_thresh)
        
        sig_trial = Sig_raw.where((start < Sig_raw["ts"]) & (Sig_raw["ts"] < end), drop=True).compute()
        rate_task.append(sig_trial.mean(dim='frame'))    
        
    trial_idx = np.arange(TFC_proto['trials_resp'])
    rate_task = xr.concat(rate_task, dim='trials').assign_coords({'trials': trial_idx}).rename("rate")
    rate_task = rate_task.mean(dim='trials').compute()
    modulation_index = (rate_task - rate_baseline) / (rate_task + rate_baseline)
    
    flag_rate = modulation_index > modu_index_thresh
    flag_peak = num_peaks > min_active_trials
    flag_modulation = flag_rate & flag_peak
    return  flag_peak #rate_baseline, rate_task


def MI_get_protocol_indices(flag_bin, TFC_proto, animal_batch, bin_width, ts_start):
    """
    Generates indices for frame-wise or binned data(e.g. 1s) with tone, trace, shock, post-shock
    flag_bin :  0: frame-wise--return timepoint ms; 1 : binning, return bin index
    Protocol: 240s Base -> [20 Tone, 20 Trace, 3 Shock, 20 Post, 97 ITI] x 6
    """
    if animal_batch == 1: # for the first batch in CA1-C and CA1-E          
        du_iti = TFC_proto['du_iti_batch1'] 
    elif animal_batch == 2:
        du_iti = TFC_proto['du_iti_batch2'] 
    else:
        print('Error of batch labels!')    
        #return xr.Dataset()   
    if flag_bin==0:   # Frame-wise
        sess_start = ts_start
        baseline_dur = TFC_proto['du_baseline']
        trial_dur = TFC_proto['du_cue'] + TFC_proto['du_trace'] + TFC_proto['du_shock'] + du_iti
        # Trial Sub-components
        tone_dur = TFC_proto['du_cue']
        trace_dur = TFC_proto['du_trace'] 
        shock_dur = TFC_proto['du_shock']
        post_shock1_dur = 10*1000 # set post-shock1 (10s), post-shock2 (10)
    elif flag_bin==1:   # with binning
        sess_start = 0
        baseline_dur = int(TFC_proto['du_baseline']/bin_width)
        trial_dur = int((TFC_proto['du_cue'] + TFC_proto['du_trace'] + TFC_proto['du_shock'] + du_iti)/bin_width)
        # Trial Sub-components
        tone_dur = int(TFC_proto['du_cue']/bin_width)
        trace_dur = int(TFC_proto['du_trace']/bin_width)
        shock_dur = int(TFC_proto['du_shock']/bin_width)
        post_shock1_dur = int(10*1000/bin_width) # set post-shock1 (10s), post-shock2 (10)
        
    # 2. Calculate Tone Onsets (Start of each trial)
    tone_onsets = []
    trace_onsets = []
    shock_onsets = []
    p_shock1_onsets = []
    p_shock2_onsets = []
    for i in range(TFC_proto['trials_resp']):
        tone_onsets.append(sess_start + baseline_dur + (i * trial_dur))
        trace_onsets.append(sess_start + baseline_dur + (i * trial_dur)+tone_dur)
        shock_onsets.append(sess_start + baseline_dur + (i * trial_dur)+tone_dur+trace_dur)
        p_shock1_onsets.append(sess_start + baseline_dur + (i * trial_dur)+tone_dur+trace_dur+shock_dur)
        p_shock2_onsets.append(sess_start + baseline_dur + (i * trial_dur)+tone_dur+trace_dur+shock_dur + post_shock1_dur)
    event_onsets = {'baseline': sess_start, 'tone': np.array(tone_onsets), 'trace':np.array(trace_onsets), 
                    'shock':np.array(shock_onsets), 'p_shock1':np.array(p_shock1_onsets),'p_shock2':np.array(p_shock2_onsets)}

    # 3. Get the saft indices for bootstrapping
    #Start with the initial 240s baseline (0 to the first tone), only use 20s pre-tone
    bs_start = event_onsets['tone'][0] - int(20*1000/bin_width)
    bs_end = event_onsets['tone'][0]
    safe_indices = list(range(bs_start, bs_end))    
    # Add the safe ITI periods for each trial (83s to 160s or trial end after tone onset)
    iti_s = int((TFC_proto['du_cue'] + TFC_proto['du_trace'] + TFC_proto['du_shock'] + 40*1000)/bin_width)
    iti_e = trial_dur
    for t in event_onsets['tone']:
        safe_indices.extend(range(t + iti_s, t + iti_e))        
    return event_onsets, np.array(safe_indices) #, valid_shuffle_starts
    
def MI_get_protocol_indices_recall(flag_bin, TFC_proto, animal_batch, bin_width, ts_start, flag_error_time):
    """
    Generates indices for frame-wise or binned data(e.g. 1s) with tone, trace, shock, post-shock
    flag_bin :  0: frame-wise--return timepoint ms; 1 : binning, return bin index
    Protocol: 240s Base -> [20 Tone, 20 Trace, 3 Shock, 20 Post, 97 ITI] x 6
    """
    if animal_batch == 1: # for the first batch in CA1-C and CA1-E (tone period 60s)        
        du_re_baseline = TFC_proto['du_re_baseline1'] 
    elif animal_batch == 2:
        du_re_baseline = TFC_proto['du_re_baseline2']
    else:
        print('Error of batch labels!')    
        #return xr.Dataset()   
    if flag_bin==0:   # Frame-wise
        sess_start = ts_start
        baseline_dur = du_re_baseline
        trial_dur = TFC_proto['du_re_cue2'] + TFC_proto['du_re_iti']
        # Trial Sub-components
        tone_dur = TFC_proto['du_re_cue2']
        post_tone_dur = 20*1000 # set post-tone (20s)
    elif flag_bin==1:   # with binning
        sess_start = 0
        baseline_dur = int(du_re_baseline/bin_width)
        trial_dur = int((TFC_proto['du_re_cue2'] + TFC_proto['du_re_iti'])/bin_width)
        # Trial Sub-components
        tone_dur = int(TFC_proto['du_re_cue2']/bin_width)
        post_tone_dur = int(20*1000/bin_width) # set post-tone (20s)        
    # 2. Calculate Tone Onsets (Start of each trial)
    tone_onsets = []
    p_tone_onsets = []
    for i in range(TFC_proto['re_trials_resp']):
        tone_onsets.append(sess_start + baseline_dur + (i * trial_dur))
        p_tone_onsets.append(sess_start + baseline_dur + (i * trial_dur) + tone_dur)
    event_onsets = {'baseline': sess_start, 'tone': np.array(tone_onsets), 'p_tone':np.array(p_tone_onsets)}
    # 3. Get the saft indices for bootstrapping
    #Start with the initial 240s baseline (0 to the first tone), only use 20s pre-tone
    bs_start = event_onsets['tone'][0] - int(20*1000/bin_width)
    bs_end = event_onsets['tone'][0]
    safe_indices = list(range(bs_start, bs_end))    
    # Add the safe ITI periods for each trial (83s to 160s or trial end after tone onset)
    iti_s = int((TFC_proto['du_re_cue2']+ 40*1000)/bin_width)   
    for idx, t in enumerate(event_onsets['tone']):
        iti_e = trial_dur
        if (idx==3) & flag_error_time:
            iti_e = trial_dur - int(60*1000/bin_width)
        safe_indices.extend(range(t + iti_s, t + iti_e))        
    return event_onsets, np.array(safe_indices) #, valid_shuffle_starts

 
def extract_resp_cells_bootstrapped_MI_regularizing(raw_traces, event_onsets, event_dur, tone_onsets, safe_indices, bin_width=1000, n_bootstraps=1000, alpha_percentile=95):
    """
    Identifies responsive cells by bootstrapping the regularized Modulation Index. 
                                        --regulized MI by its own activity
    raw_traces: (n_cells, n_bins) - Raw calcium traces (not Z-scored)
    event_onsets: Indices of event starts (e.g., Tone onset)
    event_dur_bins: Duration of the event in bins (e.g., 20 bins for 20s)
    tone_onsets: Used as the anchor for trials and safe-zone calculations
    alpha_percentile: 95 for standard alpha=0.05. Drop to 90 for a looser criterion.
    """
    n_cells = raw_traces.shape[0]
    n_trials = len(event_onsets)
    win_len_base = int(20*1000/bin_width)   # 20s pre-tone baseline in bins
    event_dur_bins = int(event_dur/bin_width)  # Event duration
    reg_c = np.nanmean(raw_traces, axis=1)    #  Regularizing Constant   
    
    # Global Noise Floor calculation (use whole baseline)
    #baseline_indices = range(0, tone_onsets[0])
    #background_data = raw_traces[:, baseline_indices]   
    background_data = raw_traces[:, safe_indices]  
    mu_global = np.nanmean(background_data, axis=1)
    std_global = np.nanstd(background_data, axis=1)
    noise_threshold = mu_global + (3 * std_global) + 1e-4
    
    # ---------------------------------------------------------
    # 1. BUILD THE "SAFE ZONE" POOL
    safe_data = raw_traces[:, safe_indices]
    safe_data = safe_data[:, ~np.isnan(safe_data).any(axis=0)] 
    n_safe_bins = safe_data.shape[1]  
    # ---------------------------------------------------------
    # 2. BOOTSTRAP THE NULL MI DISTRIBUTION
    # ---------------------------------------------------------
    # Adjacent (Mock Baseline, Mock Event) pairs to calculate Null MI.
    # The random anchor must leave enough room for a baseline BEFORE it, and an event AFTER it.
    mock_win_len_base  = event_dur_bins
    max_start = n_safe_bins - event_dur_bins
    min_start = mock_win_len_base    
    mock_onsets = np.random.randint(min_start, max_start, size=n_bootstraps)
    null_mis = np.zeros((n_cells, n_bootstraps))
    
    for i, mock_onset in enumerate(mock_onsets):
        # Extract adjacent windows mimicking a real trial
        mock_base = safe_data[:, mock_onset - mock_win_len_base : mock_onset]
        mock_event = safe_data[:, mock_onset : mock_onset + event_dur_bins]       
        mean_mock_base = np.mean(mock_base, axis=1)
        mean_mock_event = np.mean(mock_event, axis=1)        
        # Calculate Null MI for this random slice
        null_mis[:, i] = (mean_mock_event - mean_mock_base) / (mean_mock_event + mean_mock_base + reg_c)        
    # Get the customized MI threshold for each cell (e.g., 95th percentile)
    mi_thresholds = np.percentile(null_mis, alpha_percentile, axis=1)   
    # ---------------------------------------------------------
    # 3. TRIAL-BY-TRIAL EVALUATION
    # ---------------------------------------------------------      
    trial_responsive = np.zeros((n_cells, n_trials), dtype=bool)
    actual_mis = np.zeros((n_cells, n_trials))    
    for t in range(n_trials):
        # Extract real trial data
        base_data = raw_traces[:, tone_onsets[t] - win_len_base : tone_onsets[t]]
        event_data = raw_traces[:, event_onsets[t] : event_onsets[t] + event_dur_bins]        
        mean_base = np.mean(base_data, axis=1)
        mean_event = np.mean(event_data, axis=1)        
        # Calculate Actual MI
        actual_mi = (mean_event - mean_base) / (mean_event + mean_base + reg_c)
        actual_mis[:, t] = actual_mi        
        # Criterion 1: Actual MI beats the cell's Null MI distribution
        # Criterion 2: Activity Floor. Prevents cells with 0.000001 noise from passing.
        trial_responsive[:, t] = (actual_mi > mi_thresholds) & (event_data.max(axis=1) > noise_threshold)
   
    return trial_responsive
  

def create_raised_cosine_basis(duration, n_basis, timestep=1.0):
    """
    Creates a set of raised cosine basis functions for a GLM(default width=2* peak distance).

    Parameters:
    ----------
    duration : float
        The total duration of the period you are modeling (e.g., 20.0 seconds).
    n_basis : int
        The number of basis functions (humps) to create (e.g., 5).
    timestep : float, optional
        The time resolution of the data (e.g., 1.0 for 1s bins, 
        or 1/30 for 30fps video). Default is 1.0.

    Returns:
    -------
    basis_matrix : numpy.ndarray
        A 2D array of shape (n_timesteps, n_basis)
    """
    
    # 1. Create the time vector (x-axis)
    time_vec = np.arange(0, duration + timestep, timestep)
    n_timesteps = len(time_vec)    
    # 2. Create the output matrix
    basis_matrix = np.zeros((n_timesteps, n_basis))    
    # 3. Find the peak locations for each basis function
    # These are spaced evenly across the duration
    peak_centers = np.linspace(0, duration, n_basis)    
    # 4. Calculate the "half-width"
    # Set the width = 2* distance of the peaks
    half_width = duration/(n_basis-1)       
    # 5. Loop through each basis function and create its "hump"
    for i in range(n_basis):
        center = peak_centers[i]
        window_start = center - half_width
        window_end = center + half_width
        active_indices = np.where((time_vec >= window_start) & (time_vec <= window_end))[0]    
        window_times = time_vec[active_indices]
        
        # This is the core formula:
        # 1. Scale time to be from -pi to pi within the window
        scaled_time = (window_times - center) / half_width * np.pi        
        # 2. Apply the raised cosine (Hanning window)
        basis_values = (1 + np.cos(scaled_time)) / 2        
        # 3. Put these values into the big matrix
        basis_matrix[active_indices, i] = basis_values
        
    return basis_matrix[:-1,:] #Get rid of the closed interval
 
# for grid search of alpha
def fit_glm_with_alpha(y_raw, exog, alpha_grid, family, pen_mask):
    """
    Worker function: Tests a grid of alphas for a single cell.
    Returns the best alpha, best score, AND the full array of scores for global averaging.
    """
    best_aic = np.inf
    best_alpha = alpha_grid[0]
    all_aics = [] # New: Store the score for every alpha
    
    for alpha in alpha_grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                sm_glm = sm.GLM(y_raw, exog, family=family)
                res = sm_glm.fit_regularized(alpha=alpha, L1_wt=0.0, pen_mask=pen_mask, 
                                             refit=False, profile_scale=False, 
                                             maxiter=1000, cnvrg_tol=1e-5)
                
                predicted = res.predict(exog)
                deviance = family.deviance(y_raw, predicted)
                
                all_aics.append(deviance)
                
                if deviance < best_aic:
                    best_aic = deviance
                    best_alpha = alpha
        except:
            # If it fails to converge, append NaN so the list length matches alpha_grid
            all_aics.append(np.nan) 
            
    return best_alpha, best_aic, all_aics
    
def find_best_alpha_with_AIC(exog, spike_endog, cells_after_filter, fixed_power=1.5, n_jobs=20):
    """
    Determine the optimal Ridge alpha for the Tweedie family across a grid of values.
    Returns: DataFrame of cell-by-cell results, mean scores array, global best alpha.
    """
    alpha_grid = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  
    family = sm.families.Tweedie(var_power=fixed_power)
    num_cells = cells_after_filter.size
    
    # Standardize variables
    # Note: If exog already contains scaled continuous vars from the trial loop, 
    # this scaler will center the binary/basis regressors too, which is mathematically fine for Ridge.
    exog_scaled = StandardScaler().fit_transform(exog)
    exog_scaled = sm.add_constant(exog_scaled, prepend=False)

    best_alpha_results = {}
    start_time = time.time()

    print('Calculating Best Alpha with Regularization Grid Search...')
    pen_mask = np.ones(exog_scaled.shape[1])
    pen_mask[-1] = 0 # Do not penalize intercept
    
    results_aic = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_glm_with_alpha)(spike_endog[i], exog_scaled, alpha_grid, family, pen_mask) 
        for i in range(num_cells)
    ) 
    
    # Unpack the 3-tuple returned by the updated worker function
    optimal_alpha_all, best_aic_for_cell_all, all_aics_matrix = zip(*results_aic)
    
    # 1. Build the Cell-by-Cell DataFrame
    for i in range(num_cells):
        uid = cells_after_filter[i]
        best_alpha_results[uid] = {
            'optimal_alpha': optimal_alpha_all[i],
            'best_aic': best_aic_for_cell_all[i]
        }  
        
    df_alpha_each_cell = pd.DataFrame.from_dict(best_alpha_results, orient='index')
    
    # 2. Calculate Global Mean Scores
    # Convert tuple of lists into a 2D numpy array: shape (num_cells, len(alpha_grid))
    aics_array = np.array(all_aics_matrix)
    
    # Calculate the mean across all cells for each alpha (ignoring NaNs from failed fits)
    score_mean = np.nanmean(aics_array, axis=0)
    
    # 3. Find the Global Best Alpha
    # The index of the lowest mean deviance corresponds to the best global alpha
    best_global_idx = np.nanargmin(score_mean)
    alpha_d2_mean = alpha_grid[best_global_idx]
    
    end_time = time.time()
    print(f"Grid Search finished: Processed {num_cells} cells. Total time: {end_time - start_time:.2f} seconds")
    
    # Return the exact 3 variables expected by the extraction script
    return df_alpha_each_cell, score_mean.tolist(), alpha_d2_mean
    
def fit_single_cell_trial(y_raw, X_full, speed_col_idx, alpha, family, pen_mask, speed_zero_scaled):
    """
    Worker function to fit the Tweedie GLM, extract the speed delta, and calculate p-values.
    Wrapped in a strict warning filter for parallel execution.
    """
    sm_glm = sm.GLM(y_raw, X_full, family=family)  
    try:
        # 1. THE CATCH-ALL WARNING SILENCER
        # Everything inside this 'with' block is mathematically gagged.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # 2. Fit Penalized model
            res_penalized = sm_glm.fit_regularized(alpha=alpha, L1_wt=0.0, pen_mask=pen_mask, 
                                                   refit=False, profile_scale=False, 
                                                   maxiter=3000, cnvrg_tol=1e-6)            
            # Get the Speed Weight
            weight = res_penalized.params[speed_col_idx]
            if np.isnan(weight) or np.abs(weight) > 10.0:
                return y_raw, False # Return raw trace safely, cell is not speed tuned
            # 3. Calculate the Multiplicative Speed Factor
            # isolate how much the speed shifted from the resting state (0 cm/s equivalent)
            actual_speed_vector = X_full[:, speed_col_idx]
            speed_shift = actual_speed_vector - speed_zero_scaled            
            # The exponential isolates the exact multiplicative effect of speed
            # Strictly limit the exponent to prevent overflow (inf) or underflow (0.0)
            exponent = np.clip(weight * speed_shift, -10.0, 10.0)
            speed_factor = np.exp(exponent)
            
            # 4. Extract Residuals via Division!
            # Safely scale the raw trace to remove the movement amplification
            residual = y_raw / (speed_factor + 1e-12)
            
            is_speed_tuned = False           
            if weight > 0:
                # The unpenalized fit (which triggers Perfect Separation) is safely inside the warning block
                res_unpenalized = sm_glm.fit(maxiter=100, disp=False)
                pval = res_unpenalized.pvalues[speed_col_idx]
                if pval < 0.05:
                    is_speed_tuned = True
                    
        # Return cleanly after the 'with' block closes
        return residual, is_speed_tuned
        
    except:
        # If matrix inversion totally fails (e.g., cell is perfectly 0 for the entire 3 minutes)
        return y_raw, False 

# Cross registration for engram cell extraction
def mapping_from_recall_condi(uids_recall, mappings):
    """
    Maps a list of UIDs from a recall session to their corresponding UIDs 
    in a conditioning session using a MultiIndex pandas DataFrame.
    """
    if len(uids_recall) == 0:
        return []
    recall_col = mappings[('session', 'session1_recall')]
    condi_col = mappings[('session', 'session0_condi')]   
    uid_map = dict(zip(recall_col, condi_col))    
    output_list = [uid_map[uid] for uid in uids_recall if uid in uid_map and pd.notna(uid_map[uid])]
    return output_list
    
def process_TFC_condi_extract_residuals_glm(dpath_minian: str, dpath_beh: str, behav_start_end_ds, output_path, TFC_proto, flag_arti, flag_raw=1, n_jobs=20, bin_width=200):
    # Extract speed-regressed residuals via GLM 
    # ==========================================
    # [ PART 1: DATA PREPARATION ]
    try:
        # 01. Read in all the necessary files
        ts = pd.read_csv(os.path.join(dpath_minian, "timestamp_merged.dat"), delimiter="\t")
        ts_tmp = pd.read_csv(os.path.join(dpath_minian, "timeStamps_cal.csv"))       
        minian_ds = open_minian(os.path.join(dpath_minian,'minian'))# , backend="zarr")
        print("processing {}".format(dpath_minian))
    except:
        print("file missing under {}".format(dpath_minian))
        return xr.Dataset()
    # 02. Behavioral timestamp processing 
    animal = PureWindowsPath(dpath_minian).parts[-2]
    behav_start_end = behav_start_end_ds[animal]
    if 'session0_condi' in list(behav_start_end.keys()):
        key_condi = 'session0_condi' 
    elif 'session1_condiY' in list(behav_start_end.keys()):
        key_condi = 'session1_condiY' 
    ts_trials, error_time = process_ts_condi(ts, TFC_proto, behav_start_end, key_condi=key_condi)
    # Processed behavioral data
    behav = pd.read_csv(os.path.join(dpath_beh, "Processed_behavioral_data.csv")).rename_axis("fmCam1").reset_index()   
    print('Behavior period (ms): begin', ts_trials[0], ' end by calc:',
          ts_trials[TFC_proto['trials']+1],'; ***** error of time: ', error_time, 's *****')
    print('Cue starting time points(ms)',ts_trials[1:(TFC_proto['trials']+1)] )   
    if (np.absolute(error_time) > 0.5): # If time different is big (>0.3s)
        raise SystemExit("Stop right there! Check the timestamps!")
  
    #For conditioning period
    if behav_start_end['batch'] == 1:    
        du_iti = TFC_proto['du_iti_batch1'] 
    elif behav_start_end['batch'] == 2:
        du_iti = TFC_proto['du_iti_batch2'] 
    else:
        print('Error of batch labels!')    
        return xr.Dataset()        
    trial_dur = TFC_proto['du_cue'] + TFC_proto['du_trace'] + TFC_proto['du_shock'] + du_iti
    test_dur = TFC_proto['du_baseline']  + trial_dur * TFC_proto['trials_resp']
    frame_condi = ts_tmp["Frame Number"].values[-1] + 1 # Frames in the first condi session
    miniscope_ts = ts[ts['camNum']==0] #Dataframe of the behavirol recording

    bins_num = int(test_dur/bin_width)
    bins_idx = np.arange(0, bins_num, 1)
    bins = np.linspace(ts_trials[0], ts_trials[0] + test_dur, bins_num + 1)
    # For homecage period post-conditioning
    hc_dur = 300*1000# HC period, take 300s
    hc_bins_num = int(hc_dur/bin_width)
    hc_bins_idx = np.arange(bins_idx[-1]+1, bins_idx[-1]+1+hc_bins_num, 1) # After the conditioning
    hc_bins_fr= np.linspace(0, 20*hc_dur/1000, hc_bins_num + 1)
    frame_hc = frame_condi + int(20*hc_dur/1000) #minian_ds['C'].frame.values[-1]
    
    # 03. General filtering of cells
    A = minian_ds['A']    
    Sig = minian_ds['C'].chunk({"frame": -1})     
    #3.1 Filering of synchronize noises when mouse shakes its head
    Sig_raw = Sig.sel(frame = range(0, frame_condi))
    artifact_mask = np.zeros(frame_condi, dtype=bool)
    if flag_arti==1:
        motion_2d = minian_ds['motion'].sel(frame=range(0,frame_condi)).values
        fmap = map_ts(ts)
        fmap = fmap.merge(behav, how="left", on="fmCam1").set_index("fmCam0")
        Sig_raw = Sig_raw.assign_coords(speed=("frame", fmap['speed'][Sig_raw.coords["frame"].values])) 
        speed = Sig_raw.speed.values
        cleaned_traces, artifact_mask = remove_flash_artifacts_2d(Sig_raw.values, motion_2d, speed, motion_thresh=4.0, percentile=40, ca_mad_thresh=5.0, min_mad=0.0001)         
        Sig_raw = Sig_raw.copy(data=cleaned_traces)
    # 3.2check the eligible cells > min_maximas after filtering with z score
    Sig_raw = Sig_raw.assign_coords(ts=("frame", miniscope_ts['sysClock'][Sig_raw.coords["frame"].values])) 
    Sig_raw = Sig_raw.where((ts_trials[0] < Sig_raw["ts"]) & (Sig_raw["ts"] < (ts_trials[0] + test_dur)), drop=True).compute()
    #Only use conditioning period
    Sig_norm = Sig_raw.where((ts_trials[1] < Sig_raw["ts"]) & (Sig_raw["ts"] < (ts_trials[0] + test_dur)), drop=True).compute()
    s_mean = Sig_norm.mean(dim='frame')
    s_std = Sig_norm.std(dim='frame')
    Sig_qc_z_score = (Sig_norm-s_mean)/(s_std + 1e-3)
    Sig_z_score = (Sig_raw-s_mean)/(s_std + 1e-3)
    #3.2 Quality control: check the eligible cells (input cell ids shoulb be 0 ~ N-1)
    cells_after_filter = filter_abnormal_cells(Sig_raw.values, Sig_qc_z_score.values, thre_skew=1.5, thre_plateau=15, min_peaks=2, fps=20) #,is_skewed, is_sparse, no_plateaus, no_silent 
    cell_num_eligible = cells_after_filter.size
    print('Total num: ', A.unit_id.size, '; Num after filtering: ', cell_num_eligible)

    # 4 data selection Calculate the binned signal in the whole session (condi + HC)
    if flag_raw==1:
        Sig_condi = Sig_raw.sel(unit_id = cells_after_filter)
    elif flag_raw==2:      
        Sig = minian_ds['S'].chunk({"frame": -1})  
        Sig_raw = Sig.sel(frame = range(0, frame_condi))
        #Sig_raw = xr.where(artifact_mask, 0, Sig_raw)
        Sig_raw = Sig_raw.assign_coords(ts=("frame", miniscope_ts['sysClock'][Sig_raw.coords["frame"].values])) 
        Sig_raw = Sig_raw.where((ts_trials[0] < Sig_raw["ts"]) & (Sig_raw["ts"] < (ts_trials[0] + test_dur)), drop=True).compute()    
        Sig_condi = Sig_raw.sel(unit_id = cells_after_filter)
    # 04. Dependant variables---endogeneous data
    spike_endog = Sig_condi.groupby_bins('ts', bins).mean().rename({'ts_bins': 'bins'}).assign_coords({'bins': bins_idx})
        
    # -------------------------------------------------------------------------
    # PART 2: GLM BUILDING
    # -------------------------------------------------------------------------
    #print("Building Trial-by-Trial Regressors...")
    fs = 1000 / bin_width  
    
    base_bins = int(20 * fs)   
    cs_bins = int(TFC_proto['du_cue'] / 1000 * fs)     
    trace_bins = int(TFC_proto['du_trace'] / 1000 * fs) 
    us_bins = int(TFC_proto['du_shock'] / 1000 * fs)   
    pus_bins = int(9 * fs)     
    iti_bins = int(du_iti / 1000 * fs)
    
    trial_total_bins = base_bins + cs_bins + trace_bins + us_bins + iti_bins # GLM fitting is trial-based
    n_trials = TFC_proto['trials_resp']
    n_cells = spike_endog.shape[0]
    
    # Build Universal Stimulus Template
    n_cs_reg = 6
    n_trace_reg = 6
    n_pus_reg = 3
    
    cs_basis = create_raised_cosine_basis(cs_bins / fs, n_cs_reg, 1/fs)
    trace_basis = create_raised_cosine_basis(trace_bins / fs, n_trace_reg, 1/fs)
    pus_basis = create_raised_cosine_basis(pus_bins / fs, n_pus_reg, 1/fs)
    
    exog_template = np.zeros((trial_total_bins, n_cs_reg + n_trace_reg + 1 + n_pus_reg))
    
    idx_cs = base_bins
    exog_template[idx_cs : idx_cs + cs_bins, 0:n_cs_reg] = cs_basis
    
    idx_trace = idx_cs + cs_bins
    exog_template[idx_trace : idx_trace + trace_bins, n_cs_reg : n_cs_reg + n_trace_reg] = trace_basis
    
    idx_us = idx_trace + trace_bins
    us_col = n_cs_reg + n_trace_reg
    exog_template[idx_us : idx_us + us_bins, us_col] = 1.0
    
    idx_pus = idx_us + us_bins
    pus_start_col = us_col + 1
    exog_template[idx_pus : idx_pus + pus_bins, pus_start_col : pus_start_col + n_pus_reg] = pus_basis

    # Prepare speed tracking (ensure it's body center speed)
    behav['ts_bin'] = pd.cut(behav['ts'], bins=bins_num) 
    global_speed = behav.groupby('ts_bin', observed=True)['speed_body'].mean().ffill().values   
    # Prepare brain motion
    motion = minian_ds['motion'].chunk({"frame": -1}).sel(frame=range(frame_condi)).compute()
    motion_dis = motion.assign_coords(ts=("frame", miniscope_ts['sysClock'][motion.coords["frame"].values])) # start from frame 1 in ts  
    motion_dis = motion_dis.groupby_bins('ts', bins).mean().values
    global_motion_x = motion_dis[:, 0]
    global_motion_y = motion_dis[:, 1]
    # -------------------------------------------------------------------------
    # PART 3: TRAINING AND EXTRACTING RESIDUALS
    # -------------------------------------------------------------------------
    print("Training Tweedie GLM and Extracting Log-Link Residuals...")
    
    FIXED_POWER = 1.5
    FIXED_ALPHA = 0.0001 # Result of grid search with find_best_alpha_with_AIC() 
    family = sm.families.Tweedie(var_power=FIXED_POWER)
    # Storage Arrays
    raw_residuals_3d = np.zeros((n_trials, n_cells, trial_total_bins))
    flag_cells = np.zeros((n_cells, n_trials), dtype=bool)
    
    for t in range(n_trials):
        cs_time_ms = ts_trials[t+1]
        trial_start_ms = cs_time_ms - (20 * 1000) 
        
        bin_start = int((trial_start_ms - ts_trials[0]) / bin_width)
        bin_end = bin_start + trial_total_bins
        
        # 1. Slice Continuous Variables
        trial_speed = global_speed[bin_start:bin_end].reshape(-1, 1)
        trial_mx = global_motion_x[bin_start:bin_end].reshape(-1, 1)
        trial_my = global_motion_y[bin_start:bin_end].reshape(-1, 1)
        
        # 2. Standardize Continuous Variables
        scaler = StandardScaler()
        continuous_vars = np.hstack([trial_speed, trial_mx, trial_my])
        continuous_vars_scaled = scaler.fit_transform(continuous_vars)
        
        # --Calculate the scaled equivalent of 0 cm/s ---
        # The scaler stores the mean and std (scale) for each column.
        # Column 0 is the speed variable.
        mean_speed = scaler.mean_[0]
        std_speed = scaler.scale_[0]
        # Prevent divide by zero if the mouse literally didn't move the entire trial
        if std_speed == 0:
            speed_zero_scaled = 0.0 
        else:
            speed_zero_scaled = (0.0 - mean_speed) / std_speed
            
        # 3. Assemble Final Design Matrix
        X_trial = np.hstack([exog_template, continuous_vars_scaled])
        X_trial = sm.add_constant(X_trial, prepend=False) # Statsmodels requires explicit intercept
        
        speed_col_idx = X_trial.shape[1] - 4 # -4 because: Intercept(last), My, Mx, Speed
        
        # 4. Prepare Penalty Mask (Do not penalize intercept)
        pen_mask = np.ones(X_trial.shape[1])
        pen_mask[-1] = 0 
        
        # 5. Extract Calcium Data
        Y_trial = spike_endog.values[:, bin_start:bin_end]
        
        # 6. Parallel Fitting per Cell for this Trial
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(fit_single_cell_trial)(
                Y_trial[i], X_trial, speed_col_idx, FIXED_ALPHA, family, pen_mask, speed_zero_scaled
            ) for i in range(n_cells)
        )
        
        # Unpack results
        for i in range(n_cells):
            raw_residuals_3d[t, i, :] = results[i][0]
            flag_cells[i, t] = results[i][1]
    # -------------------------------------------------------------------------
    # PART 4: RE-Z-SCORING
    # -------------------------------------------------------------------------
    #print("Re-Z-Scoring Final Traces...")
    global_means = np.nanmean(raw_residuals_3d, axis=(0, 2), keepdims=True) #(n_trials, n_cells, trial_total_bins)
    global_stds = np.nanstd(raw_residuals_3d, axis=(0, 2), keepdims=True)
    # Prevent divide by zero for perfectly silent cells
    # NumPy broadcasting automatically handles (n_trials, n_cells, bins) / (1, n_cells, 1)
    z_scored_residuals_3d = (raw_residuals_3d - global_means) / (global_stds + 1e-3)
    
    # Extract the original cell IDs from spike_endog
    unit_ids = spike_endog.coords['unit_id'].values 
    cal_z_scored_residuals = xr.DataArray(
        data=z_scored_residuals_3d,
        dims=['trials', 'unit_id', 'bins'],
        coords={
            'animal': spike_endog.coords['animal'].values,
            'session': spike_endog.coords['session'].values,
            'trials': np.arange(n_trials),         # Trials: 0 to 5
            'unit_id': unit_ids,
            'bins': np.arange(trial_total_bins) # Local trial bins: 0 to trial_total_bins-1           
        },
        name='Sig_each_trial')
    
    if flag_raw==1:  
        cal_z_scored_residuals.to_netcdf(os.path.join(output_path, animal + "_Cal_residual_bin_trial.nc"))       
    elif flag_raw==2:  
        cal_z_scored_residuals.to_netcdf(os.path.join(output_path, animal + "_Spike_residual_bin_trial.nc")) 
        
    data_stat = {'animal': animal,
                 'Cell num after filtering': cell_num_eligible}

    for i in range(TFC_proto['trials_resp']):       
        # For each trial
        data_stat['speed_'+str(i+1)] = np.sum(flag_cells[:,i])/cell_num_eligible
    data_stat['speed_all_trials'] = np.sum(np.any(flag_cells, axis=1)) / cell_num_eligible          
    df_stat = pd.DataFrame(data=data_stat, index=[0])
    
    print("Pipeline Complete.")
    return df_stat, cal_z_scored_residuals


