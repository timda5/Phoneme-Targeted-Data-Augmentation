import re
import pandas as pd
import os
import sys

# --- Configuration ---
input_folder = 'relative_phoneme_analysis-master/kaldi_formatted_output_filtered/large3'
output_folder = 'relative_phoneme_analysis-master'
num_results_to_show = 10
group_tags = ['fr', 'nl']

# --- ADD: List of problematic utterance IDs to exclude ---
# You can add more IDs to this list if you find others.
# Make sure the ID matches exactly how it appears in your per_utt files.
PROBLEM_UTT_IDS_TO_EXCLUDE = ["N000044_fn000094"]

# --- Ensure output directory exists ---
if output_folder and not os.path.exists(output_folder):
    try:
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    except OSError as e:
        print(f"Error creating output directory {output_folder}: {e}")
        sys.exit(1)

def process_group_file(group_tag, num_results, ids_to_exclude):
    """
    Reads per_utt file, calculates errors, prints overall WER statistics,
    prints top N lowest error utterances, and saves their IDs,
    after excluding specified problematic utterances.
    """
    per_utt_file_path = os.path.join(input_folder, f'per_utt_final_{group_tag}_large3_finetuned.txt')
    output_ids_file_path = os.path.join(output_folder, f'lowest_error_utt_ids_{group_tag}_large3_finetuned.txt')

    utterance_errors = []
    print(f"\n--- Processing Group: {group_tag.upper()} ---")
    print(f"Reading file: {per_utt_file_path}")

    csid_pattern = re.compile(r'^(\S+)\s+.*#csid\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$')

    try:
        with open(per_utt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                match = csid_pattern.match(line)
                if match:
                    utt_id, correct_str, sub_str, ins_str, del_str = match.groups()
                    num_correct = int(correct_str)
                    num_sub = int(sub_str)
                    num_ins = int(ins_str)
                    num_del = int(del_str)

                    total_ref = num_correct + num_sub + num_del
                    total_errors = num_sub + num_del + num_ins
                    error_rate = 0.0

                    if total_ref > 0:
                        error_rate = (total_errors / total_ref) * 100.0
                    elif total_errors > 0:
                        error_rate = float('inf')

                    utterance_errors.append({
                        'utterance_id': utt_id, 'error_rate (%)': error_rate,
                        'correct (C)': num_correct, 'substitutions (S)': num_sub,
                        'deletions (D)': num_del, 'insertions (I)': num_ins,
                        'total_ref (C+S+D)': total_ref, 'total_errors (S+D+I)': total_errors
                    })

    except FileNotFoundError:
        print(f"Error: File not found at {per_utt_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred processing {group_tag}: {e}")
        return None

    if not utterance_errors:
        print(f"No utterances with #csid information found or parsed for group {group_tag}.")
        return None

    df_errors = pd.DataFrame(utterance_errors)

    # --- ADD: Filter out problematic utterances BEFORE any calculations ---
    initial_utt_count = len(df_errors)
    df_errors = df_errors[~df_errors['utterance_id'].isin(ids_to_exclude)]
    excluded_count = initial_utt_count - len(df_errors)
    if excluded_count > 0:
        print(f"Excluded {excluded_count} problematic utterance(s) from group {group_tag.upper()} based on the exclusion list.")
        # Optionally, list which ones were excluded from this specific group if you want more detail
        # actual_excluded_ids_this_group = [uid for uid in ids_to_exclude if uid in df_errors_before_filter['utterance_id'].tolist()]
        # if actual_excluded_ids_this_group:
        # print(f"   Specifically excluded: {', '.join(actual_excluded_ids_this_group)}")


    if df_errors.empty:
        print(f"No utterances remaining for group {group_tag.upper()} after filtering.")
        return None

    # --- Calculate and Print Overall WER Statistics (on the filtered data) ---
    print(f"\n--- Overall WER Statistics for Group: {group_tag.upper()} (after exclusions) ---")
    all_error_rates_series = df_errors['error_rate (%)']
    
    num_infinite_wers = all_error_rates_series[all_error_rates_series == float('inf')].shape[0]
    finite_error_rates = all_error_rates_series[all_error_rates_series != float('inf')]

    if not finite_error_rates.empty:
        desc = finite_error_rates.describe()
        print(f"Statistics based on {int(desc.get('count',0))} utterances with finite WERs:")
        print(f"count   {desc.get('count', 0):.2f}")
        print(f"mean    {desc.get('mean', float('nan')):.2f}%")
        print(f"std     {desc.get('std', float('nan')):.2f}%")
        print(f"min     {desc.get('min', float('nan')):.2f}%")
        print(f"25%     {desc.get('25%', float('nan')):.2f}%")
        print(f"50%     {desc.get('50%', float('nan')):.2f}%")
        print(f"75%     {desc.get('75%', float('nan')):.2f}%")
        print(f"max     {desc.get('max', float('nan')):.2f}%")
        if num_infinite_wers > 0:
            print(f"Additionally, {num_infinite_wers} utterance(s) had an infinite WER.")
        print(f"Total utterances included in stats for this group: {len(all_error_rates_series)}")
    elif num_infinite_wers > 0:
        print(f"All {num_infinite_wers} utterance(s) with error data had an infinite WER.")
        # ... (rest of infinite WER stats printing)
    else:
        print("No WER data (finite or infinite) found to calculate statistics after filtering.")

    # --- Display Top N Lowest Error Results & Save IDs (from filtered data) ---
    df_sorted_asc = df_errors.sort_values(by='error_rate (%)', ascending=True, na_position='last')

    print(f"\n--- Top {num_results} Utterances with LOWEST Error Rate ({group_tag.upper()}) ---")
    # ... (rest of display and save logic) ...
    display_cols = [
        'utterance_id', 'error_rate (%)', 'total_errors (S+D+I)', 'total_ref (C+S+D)',
        'correct (C)', 'substitutions (S)', 'deletions (D)', 'insertions (I)'
    ]
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(df_sorted_asc.head(num_results)[display_cols].to_string(index=False))

    top_n_utt_ids = df_sorted_asc.head(num_results)['utterance_id'].tolist()
    try:
        with open(output_ids_file_path, 'w', encoding='utf-8') as outfile:
            for utt_id in top_n_utt_ids:
                outfile.write(utt_id + '\n')
        print(f"\nSaved the top {len(top_n_utt_ids)} lowest error utterance IDs for {group_tag.upper()} to: {output_ids_file_path}")
        return top_n_utt_ids
    except IOError as e:
        print(f"\nError saving lowest error utterance IDs for {group_tag.upper()} to {output_ids_file_path}: {e}")
        return None


# --- Main Execution ---
all_top_ids = {}

for tag in group_tags:
    # Pass the exclusion list to the processing function
    top_ids = process_group_file(tag, num_results_to_show, PROBLEM_UTT_IDS_TO_EXCLUDE)
    if top_ids:
        all_top_ids[tag] = top_ids

# ... (rest of the summary print statements) ...
print("\n--- Summary of Saved Files ---")
if all_top_ids:
    for tag, ids in all_top_ids.items():
        print(f"Successfully processed and saved {len(ids)} top lowest error IDs for group {tag.upper()}.")
    # Add a note about exclusions if any were made globally
    if PROBLEM_UTT_IDS_TO_EXCLUDE:
        print(f"Note: The following utterance IDs were excluded from analysis if present in any group: {', '.join(PROBLEM_UTT_IDS_TO_EXCLUDE)}")
    print("\nRecommendation (for lowest error utterances):")
    print(f"1. Check the generated .txt files in the '{os.path.abspath(output_folder)}' folder (e.g., 'lowest_error_utt_ids_fr_large3_finetuned.txt').")
    print("2. Use the utterance IDs to locate corresponding .wav files.")
    print("3. These .wav files represent utterances where the ASR performed well (low WER).")
else:
    print("Processing may have failed or found no data for one or more groups. Please check error messages above.")

print("\nScript finished.")