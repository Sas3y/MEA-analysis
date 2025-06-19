# %% [markdown]
# wavelet!

# %%
# Cell 1: Setup and Configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import numba
import pywt
import math
import warnings

print("Libraries imported.")

# <<< Centralized configuration for easy modification >>>
# --- EDIT THESE THREE VARIABLES FOR A NEW EXPERIMENT ---
EXP_ID = '328'
ACTIVITY = 'Glut'
# Define only the unique parts of each file
file_specs = [
    {'condition': 'Ctrl', 'filename': 'ctrlGlut.tsv', 'color': 'k'},
    {'condition': 'FAX', 'filename': 'faxGlut.tsv', 'color': 'b'},
    {'condition': 'DANA', 'filename': 'danaGlut.tsv', 'color': 'r'},
]

# ---------------------------------------------------------

file_definitions = [
    {
        'exp_id': EXP_ID,
        'activity': ACTIVITY,
        'condition': spec['condition'],
        'color': spec['color'],
        'path': os.path.join('./data', EXP_ID, spec['filename'])
    }
    for spec in file_specs
]

# --- General Processing Parameters ---
MANUAL_MAX_HZ = 5.0
DURATION_TO_LOAD_MS = 300000
FIRING_RATE_BIN_MS = 200
FIRING_RATE_SMOOTHING_SIGMA = 2.5
ROBUST_SCALING_FACTOR = 10.0
SAMPLING_RATE_HZ = 20000.0
USE_FLOAT32 = True

# These replace the old SPIKE_THRESHOLD_FACTOR
WAVELET_WNAME = "mexh"          # Mother wavelet (Mexican Hat)
WAVELET_WID_MS = [0.5, 2.5]     # Range of expected spike widths in ms
WAVELET_NS = 8                  # Number of scales to analyze
WAVELET_THRESHOLD_FACTOR = 0.7  # Multiplier for the adaptive threshold
WAVELET_L_PARAM = -0.2          # Statistical thresholding parameter
WAVELET_OPTION = 'l'            # Use the 'l' statistical method

# --- Fixed Data Properties ---
HEADERS_PRIMARY = ["t", "Ch 21", "Ch 31", "Ch 41", "Ch 51", "Ch 61", "Ch 71", "Ch 12", "Ch 22", "Ch 32", "Ch 42", "Ch 52", "Ch 62", "Ch 72", "Ch 82", "Ch 13", "Ch 23", "Ch 33", "Ch 43", "Ch 53", "Ch 63", "Ch 73", "Ch 83", "Ch 14", "Ch 24", "Ch 34", "Ch 44", "Ch 54", "Ch 64", "Ch 74", "Ch 84", "Ch 15", "Ch 25", "Ch 35", "Ch 45", "Ch 55", "Ch 65", "Ch 75", "Ch 85", "Ch 16", "Ch 26", "Ch 36", "Ch 46", "Ch 56", "Ch 66", "Ch 76", "Ch 86", "Ch 17", "Ch 27", "Ch 37", "Ch 47", "Ch 57", "Ch 67", "Ch 77", "Ch 87", "Ch 28", "Ch 38", "Ch 48", "Ch 58", "Ch 68", "Ch 78"]
CHANNEL_NAMES_DATA = HEADERS_PRIMARY[1:]

print("Configuration set for Wavelet-based analysis.")

# %%
# Cell 2: Helper Functions (UPDATED with Wavelet Detector)

def load_and_truncate_data(filepath, duration_ms, column_names, use_float32=True):
    if not os.path.exists(filepath): print(f"Error: File not found at '{filepath}'"); return pd.DataFrame()
    print(f"Loading data from '{os.path.basename(filepath)}' up to {duration_ms / 1000} seconds...")
    start_time = time.time()
    dtype_map = {name: (np.float32 if use_float32 else np.float64) for name in column_names}
    try:
        iterator = pd.read_csv(filepath, sep='\t', header=None, names=column_names, dtype=dtype_map, engine='c', iterator=True, chunksize=100000)
        first_chunk = next(iterator); t_start = first_chunk['t'].iloc[0]; duration_to_reach = t_start + duration_ms
        chunks = [first_chunk[first_chunk['t'] <= duration_to_reach]]
        if chunks[0]['t'].iloc[-1] < duration_to_reach:
            for chunk in iterator:
                chunks.append(chunk[chunk['t'] <= duration_to_reach])
                if chunk['t'].iloc[-1] >= duration_to_reach: break
        df = pd.concat(chunks, ignore_index=True); df['t'] = df['t'] - t_start
        print(f"Data loaded and truncated in {time.time() - start_time:.2f} seconds. Shape: {df.shape}")
        return df
    except Exception as e: print(f"An unexpected error occurred during loading: {e}"); return pd.DataFrame()

def _parse_spike_indices(index_vector, sf_khz, wid_ms):
    """Helper function for detect_spikes_wavelet. Merges and filters spike candidates."""
    if not np.any(index_vector): return np.array([], dtype=int)
    refractory_period_samples = int(1.5 * wid_ms[1] * sf_khz); merge_period_samples = int(np.mean(wid_ms) * sf_khz)
    diff_vec = np.diff(index_vector.astype(int), prepend=0, append=0); starts = np.where(diff_vec == 1)[0]; ends = np.where(diff_vec == -1)[0] - 1
    if len(starts) == 0: return np.array([], dtype=int)
    spike_candidates = sorted(list(set([int(np.mean([s, e])) for s, e in zip(starts, ends)])))
    if len(spike_candidates) < 2: return np.array(spike_candidates, dtype=int)
    final_spikes = []; i = 0
    while i < len(spike_candidates):
        j = i + 1
        while j < len(spike_candidates) and (spike_candidates[j] - spike_candidates[i]) <= merge_period_samples: j += 1
        merged_spike = int(np.mean(spike_candidates[i:j]))
        if not final_spikes or (merged_spike - final_spikes[-1]) >= refractory_period_samples: final_spikes.append(merged_spike)
        i = j
    return np.array(final_spikes, dtype=int)

def detect_spikes_wavelet(signal, sf, wid, ns, option, l_param, wname, wavelet_threshold_factor):
    """
    Detects spikes using continuous wavelet transform (CWT).
    This is a more advanced method than simple thresholding.
    """
    sf_khz = sf / 1000.0; signal = signal - np.mean(signal); nt = len(signal)
    min_scale = wid[0] * sf_khz; max_scale = wid[1] * sf_khz
    scales = np.linspace(min_scale, max_scale, ns)
    if np.any(scales <= 1): scales[scales <= 1] = 1.1; warnings.warn(f"Some scales were <= 1 and were adjusted.")
    coeffs, _ = pywt.cwt(signal, scales, wname, sampling_period=1.0)
    l_max = 36.7368; l_val = l_param * l_max
    spike_indicator = np.zeros(nt, dtype=bool)
    for i in range(ns):
        step = int(np.round(scales[i])) if np.round(scales[i]) > 0 else 1
        n_coeffs = len(coeffs[i, :]); sigma_j = np.median(np.abs(coeffs[i, ::step])) / 0.6745
        th_j = sigma_j * np.sqrt(2 * np.log(nt)); index = np.where(np.abs(coeffs[i, :]) > th_j)[0]
        statistical_threshold = 0
        if len(index) == 0 and option == 'l': mj = th_j; ps = 1 / nt; pn = 1 - ps
        elif len(index) > 0: mj = np.mean(np.abs(coeffs[i, index])); ps = len(index) / nt; pn = 1 - ps
        if 'ps' in locals() and ps > 0 and (1 - ps) > 0 and 'mj' in locals() and mj > 0: statistical_threshold = mj/2 + (sigma_j**2 / mj) * (l_val + np.log((1-ps) / ps))
        adaptive_direct_threshold = wavelet_threshold_factor * sigma_j * math.sqrt(2 * math.log(n_coeffs))
        final_threshold = min(statistical_threshold, adaptive_direct_threshold) if statistical_threshold > 0 else adaptive_direct_threshold
        if final_threshold > 0: spike_indicator[np.where(np.abs(coeffs[i, :]) > final_threshold)[0]] = True
    return _parse_spike_indices(spike_indicator, sf_khz, wid)


def calculate_smoothed_firing_rate(spike_times_ms, total_duration_ms, bin_ms, smoothing_sigma):
    """Calculates a smoothed firing rate from a list of spike times."""
    if len(spike_times_ms) < 2:
        num_bins = int(total_duration_ms / bin_ms) + 1; time_axis = np.linspace(0, total_duration_ms / 1000.0, num_bins); return time_axis, np.zeros(num_bins)
    bins = np.arange(0, total_duration_ms + bin_ms, bin_ms); counts, _ = np.histogram(spike_times_ms, bins=bins); rate = counts / (bin_ms / 1000.0)
    # Use gaussian_filter1d from scipy for smoothing
    from scipy.ndimage import gaussian_filter1d
    smoothed_rate = gaussian_filter1d(rate.astype(float), sigma=smoothing_sigma); time_axis = (bins[:-1] + bins[1:]) / 2.0 / 1000.0
    if len(time_axis) > len(smoothed_rate): time_axis = time_axis[:len(smoothed_rate)]
    elif len(smoothed_rate) > len(time_axis): smoothed_rate = smoothed_rate[:len(time_axis)]
    return time_axis, smoothed_rate

# %%
# Cell 3: The Polished, Final Plotting Function
def get_nice_scale_bar_value(max_val):
    if max_val <= 0: return 1
    power = 10**np.floor(np.log10(max_val)); scaled_max = max_val / power
    if scaled_max < 2: return 1 * power
    if scaled_max < 5: return 2 * power
    return 5 * power

def plot_combined_activity(all_data, exp_id, activity, manual_max_hz=None):
    VERTICAL_SPACING = 1.5
    conditions = list(all_data.keys()); num_conditions = len(conditions)
    num_channels = len(CHANNEL_NAMES_DATA)
    fig_height = num_channels * VERTICAL_SPACING * 0.5
    fig, axes = plt.subplots(1, num_conditions, figsize=(10 * num_conditions, fig_height), sharey=True)
    if num_conditions == 1: axes = [axes]

    if manual_max_hz is not None and manual_max_hz > 0:
        global_visual_max_rate = manual_max_hz
        print(f"\nUsing MANUALLY SET Visual Max Rate of: {global_visual_max_rate:.2f} Hz")
    else:
        print("\nUsing AUTOMATIC scaling for Visual Max Rate.")
        all_rates_combined = np.concatenate([d['smoothed_rate'] for c in all_data.values() for d in c['activity'].values() if 'smoothed_rate' in d and len(d['smoothed_rate']) > 0])
        if len(all_rates_combined) > 0:
            data_median = np.median(all_rates_combined); mad = np.median(np.abs(all_rates_combined - data_median))
            global_visual_max_rate = data_median + ROBUST_SCALING_FACTOR * mad * 1.4826
        else: global_visual_max_rate = 1.0
        if global_visual_max_rate < 1.0: global_visual_max_rate = 1.0
    
    print(f"Generating combined plot with a shared Visual Max Rate of: {global_visual_max_rate:.2f} Hz")

    max_time = 0
    for i, condition in enumerate(conditions):
        ax = axes[i]; data = all_data[condition]; activity_data = data['activity']; details = data['details']; color = details['color']
        for j, channel_name in enumerate(CHANNEL_NAMES_DATA):
            if channel_name not in activity_data: continue
            time_axis, trace = activity_data[channel_name]['time_axis'], activity_data[channel_name]['smoothed_rate']
            if len(time_axis) == 0: continue
            if time_axis[-1] > max_time: max_time = time_axis[-1]

            clipped_indices = np.where(trace > global_visual_max_rate)[0]
            display_trace = np.clip(trace, 0, global_visual_max_rate)
            normalized_trace = display_trace / global_visual_max_rate
            
            vertical_offset = (num_channels - 1 - j) * VERTICAL_SPACING
            
            ax.plot(time_axis, normalized_trace + vertical_offset, color=color, linewidth=1.0)
            
            if len(clipped_indices) > 0:
                marker_y_position = vertical_offset + 1.0 
                ax.plot(time_axis[clipped_indices], np.full_like(clipped_indices, marker_y_position, dtype=float), 
                        marker='v', color=color, linestyle='none', markersize=3, alpha=0.8, zorder=10)
            
            if i == 0: ax.text(- (max_time * 0.02), vertical_offset + 0.5, channel_name, ha='right', va='center', fontsize=9)

        ax.set_title(details['condition'], fontsize=22, pad=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel("Time (s)", fontsize=14); ax.set_xlim(0, max_time); ax.tick_params(axis='x', labelsize=12)
        ax.set_yticks([])
        if i > 0: ax.spines['left'].set_visible(False)
        
    scale_bar_hz = get_nice_scale_bar_value(global_visual_max_rate / 2)
    print(f"Using dynamic scale bar height of: {scale_bar_hz} Hz")
    scale_bar_seconds = 30
    
    scale_y_pos = -2 * VERTICAL_SPACING 
    scale_x_pos = max_time * 0.95 - scale_bar_seconds
    scale_bar_height_norm = scale_bar_hz / global_visual_max_rate
    ax_last = axes[-1]
    
    ax_last.plot([scale_x_pos, scale_x_pos + scale_bar_seconds], [scale_y_pos, scale_y_pos], color='k', linewidth=2, clip_on=False)
    ax_last.plot([scale_x_pos, scale_x_pos], [scale_y_pos, scale_y_pos + scale_bar_height_norm], color='k', linewidth=2, clip_on=False)
    
    ax_last.text(scale_x_pos + scale_bar_seconds / 2, scale_y_pos - (0.3*VERTICAL_SPACING), f"{scale_bar_seconds} s", ha='center', va='top', fontsize=12, clip_on=False)
    ax_last.text(scale_x_pos - max_time * 0.01, scale_y_pos + scale_bar_height_norm / 2, f"{scale_bar_hz} Hz", ha='right', va='center', fontsize=12, clip_on=False)
    
    fig.suptitle(f"#{exp_id} {activity} network activity comparison", fontsize=28, y=0.98)
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.93])

    output_dir = os.path.join('graphs', exp_id); os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f"{exp_id}_{activity}_final_plot.png")
    plt.savefig(output_filepath, dpi=300)
    print(f"\nFinal polished plot saved to '{output_filepath}'"); plt.show()

# %%
# Cell 4: Main Execution Block (MODIFIED to use Wavelet Detector)
all_processed_data = {}
for details in file_definitions:
    print(f"\n{'='*60}\nPROCESSING: Exp {details['exp_id']} - {details['condition']} ({details['activity']})\n{'='*60}")
    df_data = load_and_truncate_data(filepath=details['path'], duration_ms=DURATION_TO_LOAD_MS, column_names=HEADERS_PRIMARY)
    if df_data.empty: print(f"Skipping as no data was loaded."); continue
    total_duration_ms = df_data['t'].iloc[-1] if not df_data.empty else DURATION_TO_LOAD_MS
    
    print("Detecting spikes with WAVELET method and calculating firing rates...")
    activity_data_for_condition = {}
    for channel_name in CHANNEL_NAMES_DATA:
        voltage = df_data[channel_name].values
        
        spike_indices = detect_spikes_wavelet(
            signal=voltage,
            sf=SAMPLING_RATE_HZ,
            wid=WAVELET_WID_MS,
            ns=WAVELET_NS,
            option=WAVELET_OPTION,
            l_param=WAVELET_L_PARAM,
            wname=WAVELET_WNAME,
            wavelet_threshold_factor=WAVELET_THRESHOLD_FACTOR
        )
        
        spike_times_ms = df_data['t'].iloc[spike_indices].values
        
        time_axis, smoothed_rate = calculate_smoothed_firing_rate(
            spike_times_ms=spike_times_ms,
            total_duration_ms=total_duration_ms,
            bin_ms=FIRING_RATE_BIN_MS,
            smoothing_sigma=FIRING_RATE_SMOOTHING_SIGMA
        )
        
        activity_data_for_condition[channel_name] = {'time_axis': time_axis, 'smoothed_rate': smoothed_rate}
        
    all_processed_data[details['condition']] = {'activity': activity_data_for_condition, 'details': details}

if all_processed_data:
    print("\nAll processing complete. Generating final polished plot...")
    plot_combined_activity(all_processed_data, EXP_ID, ACTIVITY, manual_max_hz=MANUAL_MAX_HZ)
else:
    print("\nNo data was processed. No plot to generate.")