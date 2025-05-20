import os
import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml 
from typing import Dict, List, Tuple, Optional
try:
    from corpus import WERDetails, AlternativeCMUDict
    from utils import HParam
except ImportError as e:
    print(f"Error importing from relative_phoneme_analysis library: {e}")
    print("Please ensure the library is installed or accessible in your Python path.")
    sys.exit(1)

# Try importing espeak-ng, handle if not found
try:
    import espeakng
except ImportError:
    print("Warning: espeakng library not found. OOV words won't be converted to phonemes.")
    espeakng = None

# --- Define the MyWERDetails class correctly ---
class MyWERDetails(WERDetails):

    def __init__(self, location, skip_calculation=False, config=None):
        self.config = config
     
        self.g2p = None
        if espeakng:
            try:
                phoneme_config = getattr(config, 'phoneme', None)
                lang_code = getattr(phoneme_config, 'espeak_lang', 'nl')
                self.g2p = espeakng.Speaker(lang_code)
                print(f"Initialized espeak-ng for language: {lang_code}")
            except Exception as e:
                print(f"Warning: Failed to initialize espeak-ng Speaker: {e}")
                self.g2p = None
        else:
             print("espeak-ng not available.")

        super().__init__(location, skip_calculation, config)


    def word_to_phoneme(self, word: str, stress_cleaned: bool) -> List[str]:
        """
        Convert a word to its phoneme representation.
        Uses the dictionary first (via parent class logic), then espeak-ng (if available) for OOV words.
        """
        is_oov = True 
        cmudict_obj = getattr(self, 'cmudict', None)

        if cmudict_obj:
             if isinstance(cmudict_obj, AlternativeCMUDict):
                 lookup_result = cmudict_obj.get_arpabet(word)
                 # Check if the result is different from the original word (case-insensitive compare might be safer)
                 # Or check if it contains the expected formatting like '{'
                 is_oov = "{" not in lookup_result 
             elif isinstance(cmudict_obj, dict):
                 is_oov = word.upper() not in cmudict_obj or not cmudict_obj[word.upper()]
             else:
                 # If cmudict exists but is an unexpected type, assume OOV
                 is_oov = True
        else:
            # If cmudict doesn't exist at all, definitely OOV
            is_oov = True


        if not is_oov:
             # Call the parent's method if not OOV
             return super().word_to_phoneme(word, stress_cleaned)

        # Handle OOV with espeak-ng if available
        if self.g2p:
            try:
                # Request IPA output from espeak-ng
                phonemes_str = self.g2p.text_to_phonemes(word, ipa=1) # ipa=1 for IPA symbols
                # Clean the IPA output (remove stress, etc.)
                # Pass False for stress_remove as this cleaner handles espeak output
                cleaned_phonemes = self.arpabet_cleaner(phonemes_str, stress_remove=False)
                if cleaned_phonemes:
                    return cleaned_phonemes
                else:
                    return [] # Return empty list if espeak gives nothing
            except Exception as e:
                 print(f"Warning: espeak-ng error for '{word}': {e}")
                 return [] # Return empty list on error
        else:
            # If espeak-ng is not available, return empty list for OOV
            return []

    # --- Modify the signature of arpabet_cleaner ---
    def arpabet_cleaner(self, phoneme_str: str, stress_remove: bool = False) -> List[str]:
        """
        Basic cleaner for phoneme strings.
        Handles cleaning espeak-ng IPA output or potentially ARPABET from dict.
        """
        if not phoneme_str: return [] # Handle empty input

        # --- IPA Cleaning (Specific to espeak-ng output) --
        cleaned_str = phoneme_str.replace("ˈ", "").replace("ˌ", "").replace("_", " ")

        # Split into potential phonemes
        phonemes = cleaned_str.split()

        # Apply ARPABET-style stress removal if requested (less likely needed for IPA)
        if stress_remove:
             phonemes = [p.rstrip('0123456789') for p in phonemes]

        # Return the cleaned (and potentially converted) phonemes
        # Filter out any empty strings that might result from splitting/cleaning
        return [p for p in phonemes if p]

# ... (analyze_phoneme_differences definition) ...
def analyze_phoneme_differences(nl_wer_path: str, fr_wer_path: str, config_path: str) -> Optional[pd.DataFrame]:
    """
    Analyze phoneme error differences between Dutch and French speakers.
    Uses MyWERDetails for dictionary/G2P handling.
    Focuses on phonemes with high frequency and high error difference.
    """
    try:
        config = HParam(config_path)
        # Store config path for relative dictionary loading
        config.config_path = config_path
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return None

    print(f"Loading NL WER details from: {nl_wer_path}")
    nl_details = MyWERDetails(nl_wer_path, config=config)
    # Check if data processing was successful (self.data is not None)
    if getattr(nl_details, 'data', None) is None:
        print(f"Error: No data processed from {nl_wer_path}. Check previous errors.")
        return None
    # Check if phoneme lists were populated
    if not hasattr(nl_details, 'all_ref_phonemes') or not nl_details.all_ref_phonemes:
         print(f"Error: No phonemes extracted from {nl_wer_path}. Check dictionary/OOV issues.")
         return None

    # Calculate PER only if not skipped during init (default is False)
    if not nl_details.skip_calculation:
        try:
            nl_phonemes, nl_pers = nl_details.all_pers()
            nl_phoneme_labels, nl_counts = nl_details.phoneme_count()
            nl_per_dict = dict(zip(nl_phonemes, nl_pers))
            nl_count_dict = dict(zip(nl_phoneme_labels, nl_counts))
            # Ensure total_nl_phonemes is calculated correctly
            total_nl_phonemes = sum(nl_count_dict.values()) if nl_count_dict else 0
            if total_nl_phonemes == 0:
                 print(f"Warning: Total NL phoneme count is zero. Frequencies will be zero.")
            print(f"NL Group: Found {len(nl_phonemes)} unique phonemes, {total_nl_phonemes} total occurrences.")
        except Exception as e:
            print(f"Error calculating NL PER/Counts: {e}")
            return None # Cannot proceed without PER calculation
    else:
        print("Warning: PER calculation skipped for NL group.")
        return None # Cannot proceed without PER calculation

    print(f"\nLoading FR WER details from: {fr_wer_path}")
    fr_details = MyWERDetails(fr_wer_path, config=config)
    # Check if data processing was successful
    if getattr(fr_details, 'data', None) is None:
        print(f"Error: No data processed from {fr_wer_path}. Check previous errors.")
        return None
    # Check if phoneme lists were populated
    if not hasattr(fr_details, 'all_ref_phonemes') or not fr_details.all_ref_phonemes:
         print(f"Error: No phonemes extracted from {fr_wer_path}. Check dictionary/OOV issues.")
         return None

    if not fr_details.skip_calculation:
        try:
            fr_phonemes, fr_pers = fr_details.all_pers()
            fr_phoneme_labels, fr_counts = fr_details.phoneme_count()
            fr_per_dict = dict(zip(fr_phonemes, fr_pers))
            fr_count_dict = dict(zip(fr_phoneme_labels, fr_counts))
            total_fr_phonemes = sum(fr_count_dict.values()) if fr_count_dict else 0
            if total_fr_phonemes == 0:
                 print(f"Warning: Total FR phoneme count is zero. Frequencies will be zero.")
            print(f"FR Group: Found {len(fr_phonemes)} unique phonemes, {total_fr_phonemes} total occurrences.")
        except Exception as e:
            print(f"Error calculating FR PER/Counts: {e}")
            return None
    else:
        print("Warning: PER calculation skipped for FR group.")
        return None

    # --- Comparison Logic ---
    # Ensure phoneme lists are not empty before proceeding
    if not nl_phonemes and not fr_phonemes:
        print("Error: No phonemes found in either group. Cannot perform comparison.")
        return None

    all_phonemes = sorted(list(set(nl_phonemes) | set(fr_phonemes)))
    print(f"\nFound {len(all_phonemes)} unique phonemes across both groups.")

    data = []
    min_count_threshold = 5 # Minimum occurrences in at least one group to be considered
    max_per_threshold = 150 # Allow PER > 100, as it can happen
    import re # Ensure re is imported if not already at top level

    for phoneme in all_phonemes:
        # Skip space character or other non-phonetic symbols if necessary
        # Adjust this check based on actual symbols in your phoneme set
        if not phoneme or phoneme.isspace() or phoneme == ' ': # Check for empty string or space
            continue

        nl_error_rate = nl_per_dict.get(phoneme, np.nan) # Use NaN for missing
        fr_error_rate = fr_per_dict.get(phoneme, np.nan) # Use NaN for missing
        nl_count = nl_count_dict.get(phoneme, 0)
        fr_count = fr_count_dict.get(phoneme, 0)

        # --- Filtering ---
        # Filter if count is too low in BOTH groups
        if nl_count < min_count_threshold and fr_count < min_count_threshold:
            continue
        # Filter if PER is NaN in BOTH groups (meaning phoneme didn't appear in ref for either)
        if np.isnan(nl_error_rate) and np.isnan(fr_error_rate):
            continue

        # --- Calculations ---
        # Handle NaN in error difference calculation
        if np.isnan(fr_error_rate) or np.isnan(nl_error_rate):
            error_diff = np.nan
        else:
            error_diff = fr_error_rate - nl_error_rate # Positive means higher error for FR group

        # Ensure denominators are not zero for frequency
        nl_freq = nl_count / total_nl_phonemes if total_nl_phonemes > 0 else 0
        fr_freq = fr_count / total_fr_phonemes if total_fr_phonemes > 0 else 0
        avg_freq = (nl_freq + fr_freq) / 2

        # Handle NaN in importance score calculation
        if np.isnan(error_diff) or avg_freq <= 0:
            importance_score = np.nan # Or 0 if preferred
        else:
            importance_score = error_diff * np.sqrt(avg_freq) * 100

        data.append({
            'Phoneme': phoneme,
            'NL_PER': nl_error_rate,
            'FR_PER': fr_error_rate,
            'Error_Diff (FR-NL)': error_diff,
            'NL_Count': nl_count,
            'FR_Count': fr_count,
            'NL_Freq (%)': nl_freq * 100,
            'FR_Freq (%)': fr_freq * 100,
            'Avg_Freq (%)': avg_freq * 100,
            'Importance_Score': importance_score
        })

    print(f"\nCreated data for {len(data)} phonemes after filtering.")

    if not data:
        print("WARNING: No valid phoneme data found after filtering!")
        return pd.DataFrame(columns=[ # Return empty DataFrame with expected columns
            'Phoneme', 'NL_PER', 'FR_PER', 'Error_Diff (FR-NL)',
            'NL_Count', 'FR_Count', 'NL_Freq (%)', 'FR_Freq (%)',
            'Avg_Freq (%)', 'Importance_Score'
        ])

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by absolute importance score, handling potential NaNs
    # Use key=abs only works if no NaNs, otherwise NaNs might sort unpredictably
    # Sort NaNs last by default, then by absolute value descending
    df = df.sort_values(by='Importance_Score', key=abs, ascending=False, na_position='last')

    return df

# ... (create_frequency_vs_difference_plot definition) ...
def create_frequency_vs_difference_plot(df: pd.DataFrame, output_filename="phoneme_error_analysis.png"):
    """Creates a scatter plot of Average Frequency vs. Error Difference."""
    if df.empty:
        print("Cannot create plot: DataFrame is empty.")
        return

    # Drop rows with NaN in columns needed for plotting to avoid errors
    plot_df = df.dropna(subset=['Avg_Freq (%)', 'Error_Diff (FR-NL)', 'Importance_Score']).copy()
    if plot_df.empty:
        print("Cannot create plot: No valid data points after dropping NaNs.")
        return

    plt.figure(figsize=(12, 8))

    # Scatter plot: x=Avg Freq, y=Error Diff, size=Importance, color=Error Diff
    scatter = plt.scatter(
        plot_df['Avg_Freq (%)'],
        plot_df['Error_Diff (FR-NL)'],
        s=np.abs(plot_df['Importance_Score']) * 5 + 10, # Scale size by importance (ensure positive) + base size
        c=plot_df['Error_Diff (FR-NL)'], # Color by error difference
        cmap='coolwarm', # Blue (NL higher error) to Red (FR higher error)
        alpha=0.7,
        edgecolors='w',
        linewidth=0.5
    )

    # Add labels for the top N most important phonemes (based on absolute importance)
    top_n = 15 # Number of phonemes to label
    # Use index from the original sorted DataFrame (df) but plot using plot_df coordinates
    # Make sure the index aligns or re-select top_n from plot_df
    plot_df_sorted_importance = plot_df.sort_values('Importance_Score', key=lambda x: abs(x), ascending=False)

    for idx in plot_df_sorted_importance.head(top_n).index:
        row = plot_df.loc[idx] # Get data from the NaN-filtered DataFrame
        plt.text(row['Avg_Freq (%)'], row['Error_Diff (FR-NL)'], f" {row['Phoneme']}", fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error Difference (FR_PER - NL_PER)')

    # Add horizontal line at y=0
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)

    # Labels and Title
    plt.xlabel('Average Frequency (%)')
    plt.ylabel('Phoneme Error Rate Difference (FR - NL) (%)')
    plt.title('Phoneme Error Analysis: Frequency vs. Error Difference (FR vs NL Speakers) for finetuned openai/whisper-large-v3')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Save the plot
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close() # Close the plot figure

# ... (create_important_phonemes_table definition) ...
def create_important_phonemes_table(df: pd.DataFrame, top_n=10):
    """Prints a table of the top N most important phonemes."""
    if df.empty:
        print("Cannot create table: DataFrame is empty.")
        return

    print(f"\n--- Top {top_n} Most Important Phonemes (High Freq & High |Error Diff|) ---")
    # Select relevant columns and format floats
    # Use index from sorted DataFrame (df is already sorted by importance)
    # Handle potential NaNs before formatting
    df_display = df.head(top_n)[[
        'Phoneme', 'Importance_Score', 'Error_Diff (FR-NL)',
        'FR_PER', 'NL_PER', 'Avg_Freq (%)', 'FR_Count', 'NL_Count'
    ]].copy() # Use copy to avoid SettingWithCopyWarning

    # Formatting for better readability, handle NaNs gracefully
    float_cols = ['Importance_Score', 'Error_Diff (FR-NL)', 'FR_PER', 'NL_PER', 'Avg_Freq (%)']
    for col in float_cols:
        # Use .map with a lambda to handle NaN before formatting
        df_display[col] = df_display[col].map(lambda x: f'{x:.2f}' if pd.notna(x) else 'NaN')

    print(df_display.to_string(index=False))
    print("-" * (len(df_display.columns) * 15)) # Adjust separator width


def main():
    """Main function to run the analysis."""
    print("Starting Phoneme Error Analysis...")

    # --- Configuration ---
    project_base = os.path.expanduser('~/relative_phoneme_analysis-master')
    wer_details_base = os.path.join(project_base, 'kaldi_formatted_output_filtered', 'large3') 

    output_base = project_base 

    # --- Paths for Dutch group (large-v2) ---
    nl_wer_path = os.path.join(wer_details_base, 'per_utt_final_nl_large3_finetuned.txt')
    # --- Paths for French group (large-v2) ---
    fr_wer_path = os.path.join(wer_details_base, 'per_utt_final_fr_large3_finetuned.txt')
    # --- Path for config file ---
    config_path = os.path.join(project_base, 'configs', 'dutch.yaml')
    # --- Paths for output files ---
    plot_output_path = os.path.join(output_base, 'phoneme_analysis_plot_large3_finetuned.png') 
    csv_output_path = os.path.join(output_base, 'phoneme_analysis_full_results_large3_finetuned.csv') 

    # --- Run Analysis ---
    try:
        analysis_df = analyze_phoneme_differences(nl_wer_path, fr_wer_path, config_path)

        if analysis_df is not None and not analysis_df.empty:
            print("\nAnalysis produced results. Creating outputs...")

            # --- SAVE TO CSV ---
            try:
                # Use index=False if you don't want the DataFrame index in the CSV
                analysis_df.to_csv(csv_output_path, index=False, encoding='utf-8')
                print(f"Full analysis results saved to: {csv_output_path}")
            except Exception as e:
                print(f"Error saving results to CSV: {e}")
            # --- END SAVE TO CSV ---

            # Create plot and table
            create_frequency_vs_difference_plot(analysis_df, plot_output_path)
            create_important_phonemes_table(analysis_df, top_n=10) # Keep top_n for console output

        else:
            print("\nAnalysis did not produce results or the DataFrame was empty. Cannot create outputs.")

    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred during the main analysis process: {e}")
        traceback.print_exc()

    print("\nAnalysis finished.")


if __name__ == "__main__":
    try:
        # Ensure re is imported if used in analyze_phoneme_differences
        import re
        # Make sure AlternativeCMUDict is available if needed by MyWERDetails
        from corpus import AlternativeCMUDict
    except ImportError:
        # Define a dummy class if corpus cannot be imported,
        # although this might cause issues later if MyWERDetails relies on it.
        class AlternativeCMUDict: pass
        print("Warning: Could not import AlternativeCMUDict from corpus.")
    main()
