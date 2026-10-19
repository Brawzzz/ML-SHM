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

from .GWDataset import GWDataset


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
OGW_DS                = "OGW-2" 
OGW_FILES_PREFIX       = "pc_f"
OGW_FILES_SUFIX        = "kHz"
OGW_FILES_EXTENSION    = ".h5"
OGW_FILES              = []

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def OGW_load(nb_cycles     : int  = None,
             freq_khz      : int  = 100,
             channels      : int | list[int] = 0,
             dir_path      : str  = None,
             subset        : str  = OGW_DS,
             damage_states : list = None,
             crop          : tuple = None,
             block_size    : int  = 10) -> GWDataset:

    """
    Load the Open Guided Waves (CFRP, temperature) dataset as a raw GWDataset.

    For each damage state, the timestamped cycles are read and ONE signal per cycle is kept (one frequency, one path).
    Signals are returned unscaled : normalization must be fitted on the train set, after the split.

    Damage state : state --> class
        healthy -- 'udam' 
        dameged -- 'dam_Dxx'
                   
    Expected layout : one folder per state
        - <dir_path>_udam/ 
        - <dir_path>_dam_D04/
        - <dir_path>_dam_D06/
        - ...

    Args
    ----------
    nb_cycles     : number of cycles to load per damage state -- defaults : nb_cycles = None
                    if nb_cycles = None -> all the cycles are load

    freq_khz      : (int) excitation frequency to keep -- defaults : freq_khz = 100
    channels      : (int) actuator-receiver paths indexes (0-65) -- defaults : channels = 0
    dir_path      : (str) path to the data directory -- defaults : dir_path = None 
                    if dir_path = None -> set to OGW_DS
    subset        : (str) OGW sub-dataset to load --> folder inside `dir_path` -- defaults subset = OGW_DS
                    e.g. "OGW-2", "OGW_CFRP_Temperature"

    damage_states : (list) damage positions to load -- defaults : damage_states = None
                    if damage_states = None -> set to D04, D12, D16, D24

    block_size    : (int) number of consecutive cycles per group
                    consecutive cycles are almost identical, so they must stay in the same split

    Returns:
    ----------
    raw_dataset : GWDataset oject containing 
    
        X      : signals of the selected path -- shape (n_signals, signal_length)
        y      : signal's labels
                    - 0 = healthy (damage tag == 0)
                    - 1 = damaged (damage tag > 0)

        groups : acquisition block of each signal, chronological ids (0 = oldest)
                    ids restart at 0 in each file, then are shifted to be unique across files

        meta   : signal's metadata, one row per signal -- columns : file, damage, weather, temperature

    Notes
    ----------
    A missing damage state folder is skipped (a message is printed).
    splits() needs at least 2 groups per class : lower block_size if the split fails.
    """

    #---------------------------------------------
    if dir_path is None or not os.path.isdir(dir_path):
        raise FileNotFoundError(f"[OGW] directory not found : {dir_path}")

    subset_dir = os.path.join(dir_path, subset)

    if not os.path.isdir(subset_dir):
        available = sorted(d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)))
        raise FileNotFoundError(f"[OGW] sub-dataset '{subset}' not found in {dir_path} (available : {available})")

    damage_states = damage_states or ["D04", "D12", "D16", "D24"]
    states        = [("udam", 0)] + [(f"dam_{d}", 1) for d in damage_states]

    #-------------------------
    next_group_id         = 0
    (X, y, groups, metas) = ([], [], [], [])

    for (state, label) in states:

        (M, df) = load_state(state_dir = os.path.join(subset_dir, state), 
                             nb_cycles = nb_cycles,
                             freq_khz  = freq_khz, 
                             channels  = channels)

        #---------------
        if crop is not None:
            M = M[..., crop[0]:crop[1]]

        #---------------
        if df is None:
            continue

        local_group_id = df["cycle_index"].to_numpy() // block_size

        X.append(M)
        y.append(np.full(len(M), label))

        groups.append(next_group_id + local_group_id)
        next_group_id += local_group_id.max() + 1

        metas.append(df.drop(columns=["catch", "pitch"], errors="ignore"))

    #-------------------------
    if not X or not (np.concatenate(y) == 0).any():
        raise RuntimeError("no healthy (udam) OGW data found")

    raw_dataset = GWDataset(name   = subset,
                            X      = np.vstack(X),
                            y      = np.concatenate(y),
                            groups = np.concatenate(groups),
                            meta   = pd.concat(metas, ignore_index=True))
    
    return raw_dataset

#================================================================================#
def load_state(state_dir : str, 
               nb_cycles : int = 1, 
               freq_khz  : int = 40, 
               channels  : int |list[int] = 0):

    """
    Load the OGW dataset for ONE damage state (healthy or damaged) and return
    the waveform matrix (n_measurements, n_samples) for ONE actuator-receiver path

    Args
    ----------
    state_dir : Path to the damage state directory
    nb_cycles : Number of cycles to load (None -> all)
    freq_khz  : Frequency in kHz to filter the data
    channels  : Actuator-receiver path index (0..65) to filter the data

    Returns:
    ----------
    tuple : (matrix, DataFrame)
    """

    #---------------------------------------------
    if not os.path.isdir(state_dir):
        print(f"[OGW] state not found, skipped : {state_dir}")
        return(None, None)
    
    cycles = load_cycles(state_dir = state_dir, 
                         n_cycles  = nb_cycles, 
                         freqs     = [freq_khz],
                         channels  = channels)
    
    df = concat_cycles(cycles)

    #-------------------------
    if df.empty:
        return(None, None)
    
    return stack_waveforms(df, channels, "catch"), df

#================================================================================#
def load_cycles(state_dir    : str,
                n_cycles     : int  = -1,
                freqs        : list = None,
                channels     : int | list[int] = None,
                keep_signals : bool = True,
                keep_pitch   : bool = False) -> dict:
    
    """
    Look through the timestamped folders (cycles) of ONE damage state of the OGW
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
 
    Args
    ----------
    state_dir    : path to one damage-state directory (udam or dam_Dxx)
    n_cycles     : number of timestamped folders to load -- defaults n_cycles = 1 
                   if nb_cycles > 0 or None -> loads all cycles

    freqs        : list of frequencies in kHz to keep --  defauts freqs = None
                   if freqs = None -> all freq are kept
    
    channels     : paths indexes 0 - 65 --defaults = 0

    keep_signals : store the 'catch' waveform(s) in the DataFrame -- defualt keep_signals = True 
    keep_pitch   : also store the reference 'pitch' waveform(s) -- defaults keep_pitch = False 
 
    Returns
    ----------
    cycles : dict { timestamp_str : DataFrame }
                - one DataFrame per cycle, 
                - one row per frequency file

             columns : 
             (metadata) damage_state, cycle_index, timestamp, freq_khz,
             temperature, temp_probe1/2, temp_chamber, humidity, fs,
             signal_freq, burst_cycles, n_paths, n_samples, file
             (+ signals) catch [, pitch] [, channels | channel_pair]
    """
 
    #---------------------------------------------
    if not os.path.isdir(state_dir):
        raise NotADirectoryError(f"state_dir not found : {state_dir}")
    
    damage_state = os.path.basename(os.path.normpath(state_dir))
    subdirs      = tools.sort_timestamp_data(state_dir=state_dir, n_cycles=n_cycles)
    
    #------------------------------
    cycles = {} 
    pbar   = tqdm.tqdm(subdirs, desc=f"Loading OGW {damage_state} cycles", unit="cycle")

    for cyc_idx, sub in enumerate(pbar):

        rows     = []
        ts_name  = os.path.basename(sub)
        h5_files = sorted(glob.glob(os.path.join(sub, "*.h5")))

        #------------------------------
        for fp in h5_files:
 
            freq_match = re.search(r"f(\d+)kHz", os.path.basename(fp))
            freq_khz   = int(freq_match.group(1)) if freq_match else np.nan
 
            if (freqs is not None) and (freq_khz not in freqs):
                continue
 
            #---------------
            with h5py.File(fp, "r") as f:
 
                catch  = f["pitchcatch/catch"][:]
                chans  = f["command/pitchcatch/channels"][:]
                fs     = float(f["command/pitchcatch/sampling_frequency"][0])
                fc     = float(f["command/pitchcatch/signal_frequency"][0])
                nburst = int(f["command/pitchcatch/signal_cycles"][0])
                T_vals = np.asarray(f["Temperature/values"][:], dtype=float)
 
                hum    = float(f["CTC/Humidity"][0])     if "CTC/Humidity"    in f else np.nan
                T_cham = float(f["CTC/Temperature"][0])  if "CTC/Temperature" in f else np.nan
 
                pitch  = f["pitchcatch/pitch"][:] if keep_pitch else None
 
            (n_paths, n_samples) = catch.shape
 
            #---------------
            row = {
                "damage_state" : damage_state,
                "cycle_index"  : cyc_idx,
                "timestamp"    : ts_name,
                "freq_khz"     : freq_khz,
                "temperature"  : float(np.mean(T_vals)),
                "temp_probe1"  : float(T_vals[0]) if T_vals.size > 0 else np.nan,
                "temp_probe2"  : float(T_vals[1]) if T_vals.size > 1 else np.nan,
                "temp_chamber" : T_cham,
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

                if channels is None:

                    row["catch"]    = catch
                    row["channels"] = chans

                    if keep_pitch:
                        row["pitch"] = pitch

                else:

                    ch = np.atleast_1d(channels)

                    row["catch"]        = catch[ch]
                    row["channel_pair"] = [tuple(int(x) for x in chans[c]) for c in ch]

                    if keep_pitch:
                        row["pitch"] = pitch[ch]
 
            rows.append(row)
 
        #------------------------------ 
        df = pd.DataFrame(rows)
        if "freq_khz" in df:
            df = df.sort_values("freq_khz").reset_index(drop=True)
 
        cycles[ts_name] = df
 
    pbar.close()
 
    return cycles

#================================================================================#
def concat_cycles(cycles: dict) -> pd.DataFrame:
 
    """
    Merge the per-cycle DataFrames returned by OGW_load() into a single
    tidy DataFrame (one row per file). 
    Convenient for a temperature sweep or to feed the validation block.
 
    Args
    ----------
    cycles : dict {timestamp : DataFrame} from load_cycles()
 
    Returns
    ----------
    df : (pd.DataFrame) concatenated DataFrame, chronologically ordered
    """
 
    #---------------------------------------------
    if not cycles:
        return pd.DataFrame()
 
    df = pd.concat(cycles.values(), ignore_index=True)
    return df.sort_values(["timestamp", "freq_khz"]).reset_index(drop=True)

#================================================================================#
def stack_waveforms(df         : pd.DataFrame, 
                    channels   : int | list[int] = 0, 
                    signal_col : str = "catch") -> np.ndarray:

    """
    Stack the waveforms stored in `df[signal_col]` into a (n_measurements, n_samples) array,
    keeping ONE actuator-receiver path.

    Each cell holds either : 
        - the full array (n_paths, n_samples) -- loaded with channel=None
        - or a single path (n_samples,) already selected at loading (channel=<int>).

    Args
    ----------
    df         : (pd.DataFrame) DataFrame from OGW_load()[ts] or concat_cycles()
    channels   : (int | list[int]) 
    signal_col : (str) 'catch' (received) or 'pitch' (reference)

    Returns
    ----------
    waveforms : (np.ndarray) waveforms of `df[signal_col] for one actuator-receiver path
    """

    #---------------------------------------------
    if channels is None:
      return np.stack([np.atleast_2d(np.asarray(c)) for c in df[signal_col].values])
    
    if df["freq_khz"].nunique() > 1:
        raise ValueError(f"several frequencies in df : {sorted(df['freq_khz'].unique())} -- filter first")

    channels = np.atleast_1d(channels)
    sigs     = []

    #--------------------
    for (i, cell) in enumerate(df[signal_col].values):

        arr = np.atleast_2d(np.asarray(cell))

        if arr.shape[0] != len(channels):              
            arr = arr[channels]

        sigs.append(arr)

    #--------------------
    if len({s.shape for s in sigs}) > 1:
        raise ValueError(f"waveforms of different shapes : {sorted({s.shape for s in sigs})}")

    return np.stack(sigs)

#================================================================================#
def OGW_plot(df          : pd.DataFrame,
             channels    : int | list[int] = 0,
             freq_khz    : int = None,
             signal_col  : str = "catch",
             n_waveforms : int = 8,
             cmap_scan   : str = "RdBu_r",
             cmap_temp   : str = "coolwarm",
             save_path   : str = None) -> None:
 
    """
    Graphical overview of the OGW guided-wave data for ONE actuator-receiver path
    and ONE excitation frequency, focused on the temperature effect.
 
    Four panels :

      1. catch waveforms overlaid, colored by temperature (time-of-flight shift)
      2. B-scan : amplitude heatmap over (time x temperature) - the temperature
         shift of the wave packet, the key SHM view (diverging colormap, 0-centered)
      3. thermal profile of the loaded measurements (acquisition order)
      4. signal RMS vs temperature (scalar-feature drift)
 
    Args
    ----------
    df           : (pd.DataFrame) DataFrame from OGW_load()[ts] or concat_cycles()
    channels      : (int) path index 0 - 65 (ignored if signals were stored single-path)
    freq_khz     : (int) frequency to display (None -> the first one found in df)
    signal_col   : (str) 'catch' (received) or 'pitch' (reference)
    n_waveforms  : (int) number of waveforms overlaid in panel 1
    cmap_scan    : (str) diverging colormap for the B-scan (signed amplitude)
    cmap_temp    : (str) sequential/thermal colormap encoding temperature
    save_path    : (str) if given, save the figure instead of only showing it

    Returns
    ----------
    None
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
    M           = stack_waveforms(sub, channels, signal_col)[:, 0, :]
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
        pair = f" (paire {tuple(int(x) for x in np.asarray(sub['channels'].iloc[0])[channels])})"
 
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
    axes[0, 0].set_title(f"Signaux {signal_col} — chemin {channels}{pair}, {freq_khz} kHz", fontsize=13)
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