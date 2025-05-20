import os
import pandas as pd
import numpy as np
import traceback
import sys # For sys.exit
import json # For loading individual JSON files
import glob # For finding JSON files
import matplotlib.pyplot as plt

# --- Constants ---
LANG_DUT = 'dut'
LANG_FRA = 'fra'
CHILD_MIN_AGE = 7
CHILD_MAX_AGE = 11

# --- Path Configuration ---
try:
    # Assuming the script is in a directory like /home/user/project/
    # For the diff, I'll use a placeholder like /path/to/your/project/
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    print("Warning: __file__ not defined. Using current working directory for relative paths.")
    script_dir = os.getcwd()

# Assumes the 'jasmin-data' folder is in the same directory as the script
# IMPORTANT: Update this path if your data is located elsewhere
base_data_path = os.path.join(script_dir, "jasmin-data/Data/data/meta/text")
output_base_dir = os.path.join(script_dir, 'output_codes') # Output folder for speaker codes lists and plots

# Store paths in a dictionary
paths = {
    'input': {
        # These are for the Exploration script to find speaker codes
        'nl_speakers': os.path.join(base_data_path, "nl/speakers.txt"),
        'vl_speakers': os.path.join(base_data_path, "vl/speakers.txt"),
    },
    'output': {
        'pure_dutch_children_codes': os.path.join(output_base_dir, 'pure_dutch_children_7_11_codes.txt'),
        'pure_french_children_codes': os.path.join(output_base_dir, 'pure_french_children_7_11_codes.txt'),
    }
}

# --- Configuration for locating Whisper script outputs ---
# IMPORTANT: Update these GROUP_NAMEs to match exactly what you used in your Whisper script runs
NL_WHISPER_GROUP_NAME = "PureDutchChildren_7_11_Large_3_Turbo" # Example, ensure this matches your run
FR_WHISPER_GROUP_NAME = "PureFrenchChildren_7_11_Large_3_Turbo" # Example, ensure this matches your run

# Base directory where Whisper script saves its group-specific output folders
WHISPER_OUTPUT_BASE_DIR = os.path.join(script_dir, 'whisper_transcriptions')


# --- Check if input files exist ---
input_paths_to_check = list(paths['input'].values())
missing_paths = [p for p in input_paths_to_check if not os.path.exists(p)]

if missing_paths:
    print("\nError: Required input speaker data files not found:")
    for p in missing_paths:
        print(f"- {os.path.abspath(p)}")
    print("Please check the paths and ensure the 'jasmin-data' folder structure is correct relative to the script.")
    sys.exit(1)
else:
    print("\nAll required input speaker files found.")

# --- Ensure output directory exists ---
if not os.path.exists(output_base_dir):
    try:
        os.makedirs(output_base_dir)
        print(f"Created output directory: {output_base_dir}")
    except OSError as e:
        print(f"Error creating output directory {output_base_dir}: {e}")
        sys.exit(1)

# --- Data Loading Function (Robust Version) ---
def load_data_with_delimiters(file_path, potential_delimiters=['\t', r'\s+'], encoding='ISO-8859-1', expected_cols=None):
    """Attempts to load a CSV/text file using a list of potential delimiters."""
    last_exception = None
    encodings_to_try = ['utf-8', encoding] # Try UTF-8 first

    for enc in encodings_to_try:
        for delim_raw in potential_delimiters:
            delim_repr = repr(delim_raw)
            try:
                engine = 'python' if delim_raw == r'\s+' else None
                df = pd.read_csv(file_path, sep=delim_raw, encoding=enc,
                                 engine=engine, on_bad_lines='warn', low_memory=False,
                                 skipinitialspace=True, comment='#', skip_blank_lines=True)
                if df.empty: continue
                df.columns = df.columns.str.strip()
                if expected_cols:
                    missing_cols = [col for col in expected_cols if col not in df.columns]
                    if not missing_cols:
                        first_col_name = df.columns[0]
                        if first_col_name and df[first_col_name].astype(str).str.match(r'^[NV]\d+').any():
                            print(f"Successfully loaded {os.path.basename(file_path)} with delimiter {delim_repr}, encoding '{enc}'.")
                            return df
                elif df.shape[1] > 1:
                     first_col_name = df.columns[0]
                     if first_col_name and df[first_col_name].astype(str).str.match(r'^[NV]\d+').any():
                         print(f"Successfully loaded {os.path.basename(file_path)} with delimiter {delim_repr}, encoding '{enc}' ({df.shape[1]} columns found).")
                         return df
            except pd.errors.ParserError as pe: last_exception = pe
            except Exception as e: last_exception = e
    print(f"Error: Could not successfully load file {file_path} with any specified delimiter/encoding.")
    if last_exception: print(f"Last error encountered: {last_exception}")
    return None

# --- Speaker Identification Function ---
def identify_pure_language_children(df, min_age=CHILD_MIN_AGE, max_age=CHILD_MAX_AGE):
    print(f"\n--- Identifying Pure Dutch & Pure French children (Age {min_age}-{max_age}) ---")
    if df is None or df.empty:
        print("Input DataFrame is empty. Cannot identify speakers.")
        return pd.DataFrame(), pd.DataFrame()
    required_cols = ['Age', 'HomeLanguage1', 'HomeLanguage2', 'RegionSpeaker']
    missing_req = [col for col in required_cols if col not in df.columns]
    if missing_req:
        print(f"Error: Missing required columns: {missing_req}. Cannot proceed.")
        return pd.DataFrame(), pd.DataFrame()
    df_processed = df.copy()
    df_processed['Age'] = pd.to_numeric(df_processed['Age'], errors='coerce')
    df_processed['HomeLanguage1'] = df_processed['HomeLanguage1'].astype(str).str.lower().str.strip().replace(['nan', 'none', ''], np.nan)
    df_processed['HomeLanguage2'] = df_processed['HomeLanguage2'].astype(str).str.lower().str.strip().replace(['nan', 'none', ''], np.nan)
    df_processed['RegionSpeaker'] = df_processed['RegionSpeaker'].astype(str).str.strip()
    df_child = df_processed[(df_processed['Age'] >= min_age) & (df_processed['Age'] <= max_age)].copy()
    print(f"Found {len(df_child)} speakers aged {min_age}-{max_age}.")
    if df_child.empty:
        print("No speakers found in the specified age range.")
        return pd.DataFrame(), pd.DataFrame()
    is_l1_dut = df_child['HomeLanguage1'] == LANG_DUT
    is_l1_fra = df_child['HomeLanguage1'] == LANG_FRA
    is_l2_dut = df_child['HomeLanguage2'] == LANG_DUT
    is_l2_fra = df_child['HomeLanguage2'] == LANG_FRA
    is_l2_empty = df_child['HomeLanguage2'].isna()
    pure_dutch_children = df_child[is_l1_dut & (is_l2_dut | is_l2_empty)].copy()
    pure_french_children = df_child[is_l1_fra & is_l2_fra].copy()
    print(f"Identified {len(pure_dutch_children)} Pure Dutch children (L1=dut, L2=dut/empty).")
    print(f"Identified {len(pure_french_children)} Pure French children (L1=fra, L2=fra).")
    print("-" * 70)
    return pure_dutch_children, pure_french_children

# --- Function to Save Speaker Codes ---
def save_speaker_codes(speaker_df, output_filepath):
    if speaker_df is None or speaker_df.empty:
        print(f"No speakers to save for {os.path.basename(output_filepath)}.")
        try:
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, 'w', encoding='utf-8') as f: pass
            print(f"Created empty file: {output_filepath}")
        except Exception as e: print(f"Error creating empty file {output_filepath}: {e}")
        return 0
    if 'RegionSpeaker' not in speaker_df.columns:
        print(f"Error: 'RegionSpeaker' column not found. Cannot save codes for {os.path.basename(output_filepath)}.")
        return 0
    speaker_codes = speaker_df['RegionSpeaker'].unique().tolist()
    speaker_codes.sort()
    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for code in speaker_codes: f.write(f"{code}\n")
        print(f"Saved {len(speaker_codes)} unique speaker codes to: {output_filepath}")
        return len(speaker_codes)
    except Exception as e:
        print(f"Error writing speaker codes to {output_filepath}: {e}")
        return 0

# --- Function to Calculate Total Hours from JSONs in a Directory ---
def calculate_total_hours_from_json_directory(group_directory_path):
    """
    Calculates total duration of transcribed text from .ort_segments in JSON files
    within a given directory.
    """
    total_duration_seconds = 0.0
    if not os.path.isdir(group_directory_path):
        print(f"  Warning: Directory not found for duration calculation: {group_directory_path}")
        return 0.0

    json_files = glob.glob(os.path.join(group_directory_path, "*.json"))
    if not json_files:
        print(f"  Warning: No JSON files found in {group_directory_path}")
        return 0.0

    print(f"  Calculating total text duration from {len(json_files)} JSON files in {os.path.basename(group_directory_path)}...")
    for json_file_path in json_files:
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            ort_segments = data.get('ort_segments', [])
            if not isinstance(ort_segments, list):
                # print(f"    Warning: 'ort_segments' is not a list or is missing in {os.path.basename(json_file_path)}. Skipping file.")
                continue

            for segment in ort_segments:
                if isinstance(segment, dict):
                    start_time = segment.get('start')
                    end_time = segment.get('end')
                    text = segment.get('text', "")

                    if isinstance(start_time, (int, float)) and \
                       isinstance(end_time, (int, float)) and \
                       isinstance(text, str) and text.strip(): # Check for actual text
                        
                        duration = end_time - start_time
                        if duration > 0: # Ensure positive duration
                            total_duration_seconds += duration
                # else:
                    # print(f"    Warning: Segment in {os.path.basename(json_file_path)} is not a dictionary. Segment: {segment}")
        except json.JSONDecodeError:
            print(f"    Warning: Could not decode JSON from {os.path.basename(json_file_path)}. Skipping.")
        except Exception as e:
            print(f"    Warning: Error processing file {os.path.basename(json_file_path)}: {e}")
            
    total_duration_hours = total_duration_seconds / 3600.0
    return total_duration_hours

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting Speaker Identification and Data Exploration ({CHILD_MIN_AGE}-{CHILD_MAX_AGE})...")

    expected_speaker_cols = ['RegionSpeaker', 'ResPlace', 'Gender', 'Age', 'BirthPlace', 'Group', 'HomeLanguage1', 'HomeLanguage2', 'CEF']
    nl_speakers_df = load_data_with_delimiters(paths['input']['nl_speakers'], potential_delimiters=[r'\s+', '\t'], expected_cols=expected_speaker_cols)
    vl_speakers_df = load_data_with_delimiters(paths['input']['vl_speakers'], potential_delimiters=[r'\s+', '\t'], expected_cols=expected_speaker_cols)

    combined_df = None
    loaded_dfs = [df for df in [nl_speakers_df, vl_speakers_df] if df is not None]
    if loaded_dfs:
        combined_df = pd.concat(loaded_dfs, ignore_index=True)
        print(f"\nCombined {len(loaded_dfs)} speaker file(s): {len(combined_df)} total entries.")
    else:
        print("\nError: Failed to load any speaker data. Exiting.")
        sys.exit(1)

    pure_dutch_df, pure_french_df = identify_pure_language_children(combined_df, min_age=CHILD_MIN_AGE, max_age=CHILD_MAX_AGE)

    print("\n--- Saving Speaker Codes ---")
    save_speaker_codes(pure_dutch_df, paths['output']['pure_dutch_children_codes'])
    save_speaker_codes(pure_french_df, paths['output']['pure_french_children_codes'])

    # --- Calculate Total Hours from Whisper Output Directories ---
    print("\n--- Calculating Total Text Durations from Whisper Output JSONs ---")
    nl_whisper_group_dir = os.path.join(WHISPER_OUTPUT_BASE_DIR, NL_WHISPER_GROUP_NAME)
    fr_whisper_group_dir = os.path.join(WHISPER_OUTPUT_BASE_DIR, FR_WHISPER_GROUP_NAME)

    nl_hours = calculate_total_hours_from_json_directory(nl_whisper_group_dir)
    fr_hours = calculate_total_hours_from_json_directory(fr_whisper_group_dir)
    
    print(f"Calculated NL hours (from {NL_WHISPER_GROUP_NAME} JSONs): {nl_hours:.2f}")
    print(f"Calculated FR hours (from {FR_WHISPER_GROUP_NAME} JSONs): {fr_hours:.2f}")

    num_nl_speakers = pure_dutch_df['RegionSpeaker'].nunique() if not pure_dutch_df.empty else 0
    num_fr_speakers = pure_french_df['RegionSpeaker'].nunique() if not pure_french_df.empty else 0

    print("\n--- Data Sufficiency Summary ---")
    print(f"Native Dutch-speaking children (NL):")
    print(f"  - Speakers: {num_nl_speakers}")
    print(f"  - Hours of text (calculated from .ort segments in JSONs): {nl_hours:.2f} hours")

    print(f"\nFrench-speaking children learning Dutch (FR):")
    print(f"  - Speakers: {num_fr_speakers}")
    print(f"  - Hours of text (calculated from .ort segments in JSONs): {fr_hours:.2f} hours")
    print("-" * 70)

    # --- Plotting Data Sufficiency ---
    print("\n--- Generating Data Sufficiency Plots ---")
    groups = ['Native Dutch (NL)', 'French-speaking (FR)']
    plot_colors = ['skyblue', 'lightcoral']

    # Plot 1: Number of Speakers
    speaker_counts = [num_nl_speakers, num_fr_speakers]
    plt.figure(figsize=(7, 5))
    bars_speakers = plt.bar(groups, speaker_counts, color=plot_colors)
    plt.ylabel('Number of Speakers')
    plt.title('Number of Speakers per Group')
    for bar in bars_speakers:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(speaker_counts) if max(speaker_counts) > 0 else 0.5, int(yval), va='bottom', ha='center')
    plt.tight_layout()
    speaker_plot_path = os.path.join(output_base_dir, 'speakers_per_group_calculated.png') # Added _calculated
    plt.savefig(speaker_plot_path)
    plt.close()
    print(f"Saved plot: {speaker_plot_path}")

    # Plot 2: Hours of Text (Calculated)
    hours_speech = [nl_hours, fr_hours]
    plt.figure(figsize=(7, 5))
    bars_hours = plt.bar(groups, hours_speech, color=plot_colors)
    plt.ylabel('Hours of Text (from .ort segments)')
    plt.title('Hours of Text per Group (Calculated)')
    for bar in bars_hours:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(hours_speech) if max(hours_speech) > 0 else 0.05, f'{yval:.2f}', va='bottom', ha='center')
    if nl_hours == 0 or fr_hours == 0 :
        plt.text(0.5, 0.5, 'One or both groups have 0 calculated hours.\nCheck Whisper output directories and JSON content.',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=9, color='red',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    plt.tight_layout()
    hours_plot_path = os.path.join(output_base_dir, 'hours_of_speech_per_group_calculated.png') # Added _calculated
    plt.savefig(hours_plot_path)
    plt.close()
    print(f"Saved plot: {hours_plot_path}")

    print("-" * 70)
    print("Note on Phoneme Distribution Analysis (for later stages):")
    print("This script now calculates total text hours from existing Whisper JSON outputs.")
    print("For phoneme distribution, you would still need to:")
    print("1. Convert transcriptions (reference or ASR from these JSONs) to phoneme sequences.")
    print("2. Count phoneme occurrences for each group.")
    print("3. Visualize these frequencies.")
    print("-" * 70)

    print("\nScript finished.")
