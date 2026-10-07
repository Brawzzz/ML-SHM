#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import re
import h5py
import glob
import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import tools
from .base import SHMDataset


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
OGW_FILES_PREFIX       = "pc_f"
OGW_FILES_SUFIX        = "kHz"
OGW_FILES_EXTENSION    = ".h5"

#================================================================================#
def _ogw_prefix(dir_path: str) -> str:
    """Default state-directory prefix : <dir_path>/OGW/OGW_CFRP_Temperature (+ '_udam', '_dam_D04' ...)"""
    return os.path.join(dir_path, "OGW", "OGW_CFRP_Temperature")

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def OGW_load(dir_path: str,
             nb_cycles: int = None,
             freq_khz: int = 100,
             channel: int = 0,
             base_dir: str = None,
             damage_states: list = None,
             block_size: int = 10,
             verbose: bool = True) -> SHMDataset:

    """
    Load the RAW OGW dataset (no normalisation, no split).

    File format : healthy = '<base_dir>_udam'  |  damaged = '<base_dir>_dam_D04' ... '_dam_D24'

    Consecutive cycles of one damage state are acquired at neighbouring temperatures,
    so they are grouped by blocks of `block_size` cycles : a block is never shared
    between train / val / test (otherwise neighbouring, almost identical
    measurements end up on both sides of the split).

    Parameters
    ----------
    dir_path      : root datasets directory
    nb_cycles     : timestamped folders per state (None -> all)
    freq_khz      : excitation frequency to keep
    channel       : actuator-receiver path index (0..65)
    base_dir      : state-directory prefix (default : <dir_path>/OGW/OGW_CFRP_Temperature)
    damage_states : damage positions (default : D04, D12, D16, D24)
    block_size    : number of consecutive cycles per acquisition block
    verbose       : print loading information

    Returns
    ----------
    SHMDataset : X (all signals), y (0 healthy / 1 damaged), groups (blocks of cycles), meta
    """

    #---------------------------------------------
    if base_dir is None:
        base_dir = _ogw_prefix(dir_path)

    if damage_states is None:
        damage_states = ["D04", "D12", "D16", "D24"]

    states = [("udam", 0)] + [(f"dam_{d}", 1) for d in damage_states]

    #---------------------------------------------
    (X_list, y_list, g_list, meta_list) = ([], [], [], [])
    next_block = 0

    for (state, label) in states:

        state_dir = f"{base_dir}_{state}"
        (X_s, df_s) = _load_state(state_dir, nb_cycles, freq_khz, channel)

        if df_s is None or df_s.empty:
            if label == 0:
                raise RuntimeError(f"[OGW] no healthy data found in : {state_dir} "
                                   f"(set data.params.base_dir in the experiment config if your layout differs)")
            continue

        #---------------
        df_s   = df_s.sort_values("timestamp").reset_index(drop=True)
        X_s    = _extract_matrix(df_s, channel, "catch")
        blocks = np.arange(len(df_s)) // max(1, block_size)

        X_list.append(X_s)
        y_list.append(np.full(len(df_s), label))
        g_list.append(blocks + next_block)
        meta_list.append(df_s[["damage_state", "timestamp", "temperature"]])

        next_block += int(blocks.max()) + 1

    #---------------------------------------------
    ds = SHMDataset(name   = "OGW",
                    X      = np.vstack(X_list),
                    y      = np.concatenate(y_list),
                    groups = np.concatenate(g_list),
                    meta   = pd.concat(meta_list, ignore_index=True))

    if verbose:
        print(f"\n{ds.describe()}\n")

    return ds

#================================================================================#
def OGW_cycles(state_dir: str,
             n_cycles: int = 1,
             freqs: list = None,
             channel: int = None,
             keep_signals: bool = True,
             keep_pitch: bool = False,
             verbose: bool = True) -> dict:
    
    """
    Walk through the timestamped folders (cycles) of ONE damage state of the OGW
    temperature dataset and store each measurement into pandas DataFrames.
 
    Folder layout expected :
 
        state_dir/                         <- e.g OGW_CFRP_Temperature_udam/
            20181213T095826/               <- timestamped folder = one acquisition
                pc_f40kHz.h5               <- one file per excitation frequency
                pc_f60kHz.h5
                ...
            20181213T100112/
                pc_f40kHz.h5            
                pc_f60kHz.h5
                ...
 
    Parameters
    ----------
    state_dir    : path to one damage-state directory (udam or dam_Dxx)
    n_cycles     : number of timestamped folders to load (None -> all)
    freqs        : list of frequencies in kHz to keep (None -> all, e.g. [40, 100, 260])
    channel      : path index 0..65 to keep only one actuator-receiver pair
                   (None -> keep the full (66, N) array). Strongly reduces memory.
    keep_signals : store the 'catch' waveform(s) in the DataFrame (object column)
    keep_pitch   : also store the reference 'pitch' waveform(s)
    verbose      : print progress / memory estimate
 
    Returns
    ----------
    cycles : dict { timestamp_str : DataFrame }
             one DataFrame per cycle, 
             one row per frequency file
             columns : 
             (metadata) damage_state, cycle_index, timestamp, freq_khz,
             temperature, temp_probe1/2, temp_chamber, humidity, fs,
             signal_freq, burst_cycles, n_paths, n_samples, file
             (+ signals) catch [, pitch] [, channels | channel_pair]
    """
 
    #---------------------------------------------
    if not os.path.isdir(state_dir):
        raise NotADirectoryError(f"state_dir not found : {state_dir}")
 
    damage_state    = os.path.basename(os.path.normpath(state_dir))
    subdirs         = tools.sort_timestamp_data(state_dir=state_dir, n_cycles=n_cycles)

    #------------------------------
    if verbose and keep_signals and channel is None:

        memory_estimation = tools.memory_estimate(n_subdirs=subdirs, 
                                                  n_keep_pitch=keep_pitch,
                                                  n_freqs=freqs)

    #---------------------------------------------
    print(f"\n[OGW] state '{damage_state}' | {len(subdirs)} cycle(s) to load \n")
 
    cycles  = {} 
    pbar    = tqdm.tqdm(subdirs, desc="Loading OGW cycles", unit="cycle", disable=not verbose)

    #------------------------------
    for cyc_idx, sub in enumerate(pbar):
 
        ts_name  = os.path.basename(sub)
        h5_files = sorted(glob.glob(os.path.join(sub, "*.h5")))
        rows     = []

        #------------------------------
        for fp in h5_files:
 
            freq_match = re.search(r"f(\d+)kHz", os.path.basename(fp))
            freq_khz    = int(freq_match.group(1)) if freq_match else np.nan
 
            if freqs is not None and freq_khz not in freqs:
                continue
 
            #---------------
            with h5py.File(fp, "r") as f:
 
                catch  = f["pitchcatch/catch"][:]
                chans  = f["command/pitchcatch/channels"][:]
                fs     = float(f["command/pitchcatch/sampling_frequency"][0])
                fc     = float(f["command/pitchcatch/signal_frequency"][0])
                nburst = int(f["command/pitchcatch/signal_cycles"][0])
                t_vals = np.asarray(f["Temperature/values"][:], dtype=float)
 
                hum    = float(f["CTC/Humidity"][0])     if "CTC/Humidity"    in f else np.nan
                t_cham = float(f["CTC/Temperature"][0])  if "CTC/Temperature" in f else np.nan
 
                pitch  = f["pitchcatch/pitch"][:] if keep_pitch else None
 
            (n_paths, n_samples) = catch.shape
 
            #---------------
            row = {
                "damage_state" : damage_state,
                "cycle_index"  : cyc_idx,
                "timestamp"    : ts_name,
                "freq_khz"     : freq_khz,
                "temperature"  : float(np.mean(t_vals)),
                "temp_probe1"  : float(t_vals[0]) if t_vals.size > 0 else np.nan,
                "temp_probe2"  : float(t_vals[1]) if t_vals.size > 1 else np.nan,
                "temp_chamber" : t_cham,
                "humidity"     : hum,
                "fs"           : fs,
                "signal_freq"  : fc,
                "burst_cycles" : nburst,
                "n_paths"      : n_paths,
                "n_samples"    : n_samples,
                "file"         : fp,
            }
 
            #---------------
            if keep_signals:

                if channel is None:

                    row["catch"]    = catch
                    row["channels"] = chans

                    if keep_pitch:
                        row["pitch"] = pitch

                else:
                    
                    row["catch"]        = catch[channel]
                    row["channel_pair"] = tuple(int(x) for x in chans[channel])

                    if keep_pitch:
                        row["pitch"] = pitch[channel]
 
            rows.append(row)
 
        #------------------------------ 
        df = pd.DataFrame(rows)
        if "freq_khz" in df:
            df = df.sort_values("freq_khz").reset_index(drop=True)
 
        cycles[ts_name] = df
 
    pbar.close()
 
    #---------------------------------------------
    if verbose:

        n_files = sum(len(df) for df in cycles.values())
        t_min   = min(df["temperature"].min() for df in cycles.values() if len(df))
        t_max   = max(df["temperature"].max() for df in cycles.values() if len(df))

        print(f"\n[OGW] done : {len(cycles)} cycles, {n_files} files, "
              f"T in [{t_min:.1f}, {t_max:.1f}] C \n")
 
    return cycles

#================================================================================#
def OGW_concat(cycles: dict) -> pd.DataFrame:
 
    """
    Merge the per-cycle DataFrames returned by OGW_load() into a single
    tidy DataFrame (one row per file). Convenient for a temperature sweep or to
    feed the validation block.
 
    Parameters
    ----------
    cycles : dict {timestamp : DataFrame} from OGW_cycles()
 
    Returns
    ----------
    df : concatenated DataFrame, chronologically ordered
    """
 
    #---------------------------------------------
    if not cycles:
        return pd.DataFrame()
 
    df = pd.concat(cycles.values(), ignore_index=True)
    return df.sort_values(["timestamp", "freq_khz"]).reset_index(drop=True)

#================================================================================#
def _load_state(state_dir : str, nb_cycles : int, 
                freq_khz : int, channel : int):

    """
    Load the OGW dataset for ONE damage state (healthy or damaged) and return
    the waveform matrix (n_measurements, n_samples) for ONE actuator-receiver path

    Parameters:
    ----------
    state_dir : Path to the damage state directory
    nb_cycles : Number of cycles to load (None -> all)
    freq_khz  : Frequency in kHz to filter the data
    channel   : Actuator-receiver path index (0..65) to filter the data

    Returns:
    ----------
    tuple : (matrix, DataFrame)
    """

    #---------------------------------------------
    if not os.path.isdir(state_dir):
        print(f"[OGW] state not found, skipped : {state_dir}")
        return None, None
    
    cycles = OGW_cycles(state_dir, 
                        n_cycles=nb_cycles, 
                        freqs=[freq_khz],
                        channel=channel, 
                        verbose=False)
    
    df = OGW_concat(cycles)

    #-------------------------
    if df.empty:
        return None, None
    
    return _extract_matrix(df, channel, "catch"), df

#================================================================================#
def _extract_matrix(df: pd.DataFrame, channel: int, signal_col: str):
 
    """
    Build the (n_measurements, n_samples) waveform matrix for ONE path from the
    DataFrame produced by OGW_cycles(). Handles both storage cases : full
    (66, N) cell (channel is used to index) or already-1D cell (channel ignored).

    Parameters:
    ----------
    df          : DataFrame containing the waveform data
    channel     : Actuator-receiver path index
    signal_col  : Column name containing the signal data

    Returns:
    ----------
    2D array of shape (n_measurements, n_samples) containing the waveform data
    """
 
    #---------------------------------------------
    sigs = []
    for cell in df[signal_col].values:
        arr = np.asarray(cell)
        sigs.append(arr[channel] if arr.ndim == 2 else arr)
 
    return np.vstack(sigs)

#================================================================================#
def OGW_plot(df: pd.DataFrame,
             channel: int = 0,
             freq_khz: int = None,
             signal_col: str = "catch",
             n_waveforms: int = 8,
             cmap_scan: str = "RdBu_r",
             cmap_temp: str = "coolwarm",
             save_path: str = None) -> None:
 
    """
    Graphical overview of the OGW guided-wave data for ONE actuator-receiver path
    and ONE excitation frequency, focused on the temperature effect.
 
    Four panels :

      1. catch waveforms overlaid, colored by temperature (time-of-flight shift)
      2. B-scan : amplitude heatmap over (time x temperature) - the temperature
         shift of the wave packet, the key SHM view (diverging colormap, 0-centered)
      3. thermal profile of the loaded measurements (acquisition order)
      4. signal RMS vs temperature (scalar-feature drift)
 
    Parameters
    ----------
    df           : DataFrame from OGW_load_cycles()[ts] or OGW_concat()
    channel      : path index 0..65 (ignored if signals were stored single-path)
    freq_khz     : frequency to display (None -> the first one found in df)
    signal_col   : 'catch' (received) or 'pitch' (reference)
    n_waveforms  : number of waveforms overlaid in panel 1
    cmap_scan    : diverging colormap for the B-scan (signed amplitude)
    cmap_temp    : sequential/thermal colormap encoding temperature
    save_path    : if given, save the figure instead of only showing it
    """
 
    #---------------------------------------------
    if signal_col not in df.columns:
        raise KeyError(f"'{signal_col}' not in DataFrame (load with keep_signals / keep_pitch)")
 
    #---------------------------------------------
    if freq_khz is None:
        freq_khz = int(df["freq_khz"].dropna().iloc[0])
 
    sub = df[df["freq_khz"] == freq_khz].copy()
    if sub.empty:
        print(f"[OGW] no measurement at {freq_khz} kHz in this DataFrame")
        return
 
    sub = sub.sort_values("temperature").reset_index(drop=True)
 
    #---------------------------------------------
    M           = _extract_matrix(sub, channel, signal_col)
    temps       = sub["temperature"].values
    fs          = float(sub["fs"].iloc[0])
    (n_meas, N) = M.shape
    t           = np.arange(N) / fs

    if   t[-1] < 1e-3: t_plot, unit = t * 1e6, "µs"
    elif t[-1] < 1.0:  t_plot, unit = t * 1e3, "ms"
    else:              t_plot, unit = t,        "s"
 
    pair = ""
    if "channel_pair" in sub.columns:
        pair = f" (paire {sub['channel_pair'].iloc[0]})"
    elif "channels" in sub.columns:
        pair = f" (paire {tuple(int(x) for x in np.asarray(sub['channels'].iloc[0])[channel])})"
 
    #---------------------------------------------
    sns.set_theme(style="whitegrid")

    (fig, axes )  = plt.subplots(2, 2, figsize=(16, 10))
    norm        = Normalize(vmin=temps.min(), vmax=temps.max())
    cmap        = plt.get_cmap(cmap_temp)
 
    #---------------------------------------------
    idx = np.unique(np.linspace(0, n_meas - 1, min(n_waveforms, n_meas)).astype(int))
    for i in idx:
        axes[0, 0].plot(t_plot, M[i], color=cmap(norm(temps[i])), lw=1.0, alpha=0.9)
 
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes[0, 0], label="Température (°C)")
    axes[0, 0].set_title(f"Signaux {signal_col} — chemin {channel}{pair}, {freq_khz} kHz", fontsize=13)
    axes[0, 0].set_xlabel(f"Temps ({unit})")
    axes[0, 0].set_ylabel("Amplitude")
 
    #---------------------------------------------
    if n_meas > 1:
        vmax = np.percentile(np.abs(M), 99)
        im = axes[0, 1].imshow(M, aspect="auto", origin="lower", cmap=cmap_scan,
                               vmin=-vmax, vmax=vmax,
                               extent=[t_plot[0], t_plot[-1], temps.min(), temps.max()])
        fig.colorbar(im, ax=axes[0, 1], label="Amplitude")

    else:
        axes[0, 1].text(0.5, 0.5, "B-scan : besoin de >= 2 mesures",
                        ha="center", va="center", transform=axes[0, 1].transAxes)
        
    axes[0, 1].set_title("B-scan — décalage du paquet d'onde avec la température", fontsize=13)
    axes[0, 1].set_xlabel(f"Temps ({unit})")
    axes[0, 1].set_ylabel("Température (°C)")

    #---------------------------------------------
    axes[1, 0].plot(np.arange(n_meas), temps, color="steelblue", lw=1.8, marker="o", ms=3)
    axes[1, 0].set_title("Profil thermique des mesures chargées", fontsize=13)
    axes[1, 0].set_xlabel("Mesure (ordre par température)")
    axes[1, 0].set_ylabel("Température (°C)")
 
    #--------------------------------------------- 
    rms = np.sqrt(np.mean(M ** 2, axis=1))
    axes[1, 1].scatter(temps, rms, s=18, color="darkorange", alpha=0.8, edgecolor="black", linewidth=0.3)
    axes[1, 1].set_title("Énergie du signal (RMS) vs température", fontsize=13)
    axes[1, 1].set_xlabel("Température (°C)")
    axes[1, 1].set_ylabel("RMS")
 
    #---------------------------------------------
    fig.suptitle(f"OGW — {sub['damage_state'].iloc[0]} — {n_meas} mesures", fontsize=15, fontweight="bold")
    plt.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        print(f"[OGW] figure saved -> {save_path}")
 
    plt.show()