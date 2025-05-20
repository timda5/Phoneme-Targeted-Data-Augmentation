import pandas as pd
import json
import os
import re
from pydub import AudioSegment
from tqdm import tqdm # Optional: for progress bar
import logging # Use logging instead of print for warnings

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for Speaker Groups ---

# Base directory where the script is located (assuming home directory)
base_dir = os.path.expanduser("~") # Gets /home/tdamen

# Define configurations for each group
groups_config = [
    {
        "name": "fr", # French background children
        "summary_csv": os.path.join(base_dir, 'whisper_transcriptions/PureFrenchChildren_7_11_Large3/transcription_summary_PureFrenchChildren_7_11.csv'),
        "json_dir": os.path.join(base_dir, 'whisper_transcriptions/PureFrenchChildren_7_11_Large3/'),
        "segmented_audio_dir": os.path.join(base_dir, 'espnet_data/segmented_audio_fr'),
        "espnet_data_dir": os.path.join(base_dir, 'espnet_data/train_fr') # Initial output dir
    },
    {
        "name": "nl", # Dutch background children
        "summary_csv": os.path.join(base_dir, 'whisper_transcriptions/PureDutchChildren_7_11_Large3/transcription_summary_PureDutchChildren_7_11.csv'), #<-- ADJUST IF FILENAME DIFFERS
        "json_dir": os.path.join(base_dir, 'whisper_transcriptions/PureDutchChildren_7_11_Large3/'), #<-- ADJUST IF DIRNAME DIFFERS
        "segmented_audio_dir": os.path.join(base_dir, 'espnet_data/segmented_audio_nl'),
        "espnet_data_dir": os.path.join(base_dir, 'espnet_data/train_nl') # Initial output dir
    }
]

# Base directory where the original full JASMIN WAV files are located
jasmin_audio_base_dir = os.path.join(base_dir, 'jasmin-data/Data/data/audio/wav/')

# --- Helper Functions ---

def clean_text(text):
    """
    Cleans transcription text for ESPnet:
    - Converts to uppercase.
    - Removes common punctuation (.,?!").
    - Replaces multiple spaces with a single space.
    - Strips leading/trailing whitespace.
    """
    if not isinstance(text, str):
        return "" # Return empty string if input is not text (e.g., None)
    text = text.upper()
    # Remove punctuation - keep apostrophes for now if G2P handles them
    text = re.sub(r'[.,?!"():;]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Processing Logic ---

logging.info("Starting ESPnet data preparation for all groups...")

# Loop through each group configuration
for config in groups_config:
    group_name = config["name"]
    summary_csv_path = config["summary_csv"]
    json_dir = config["json_dir"]
    segmented_audio_dir = config["segmented_audio_dir"]
    espnet_data_dir = config["espnet_data_dir"]

    logging.info(f"--- Processing Group: {group_name.upper()} ---")

    # Create output directories for the current group if they don't exist
    os.makedirs(segmented_audio_dir, exist_ok=True)
    os.makedirs(espnet_data_dir, exist_ok=True)
    logging.info(f"Output directories ensured: {segmented_audio_dir}, {espnet_data_dir}")

    # Lists to hold ESPnet file contents (reset for each group)
    wav_scp_list = []
    text_list = []
    utt2spk_list = []

    # Read the summary CSV for the current group
    try:
        summary_df = pd.read_csv(summary_csv_path)
        logging.info(f"Successfully read summary CSV: {summary_csv_path}")
    except FileNotFoundError:
        logging.error(f"Error: Summary CSV not found for group '{group_name}' at {summary_csv_path}. Skipping group.")
        continue # Skip to the next group

    logging.info(f"Processing {len(summary_df)} recordings for group '{group_name}'...")

    # Iterate through each recording in the summary file
    for index, row in tqdm(summary_df.iterrows(), total=summary_df.shape[0], desc=f"Recordings ({group_name.upper()})"):
        try: # Add a general try-except for row processing
            original_wav_filename = row['processed_wav_file']
            speaker_id = row['speaker_code']
            component = row['component'] # e.g., 'comp-q'
            file_root = row['file_root'] # e.g., 'fv170059'

            # Construct JSON filename
            json_filename = f"{speaker_id}_{component}_{file_root}.json"
            json_filepath = os.path.join(json_dir, json_filename)

            # Construct original WAV file path
            # Infer region code (nl/vl) - This might need refinement if NL speakers have different ID patterns
            region_code = 'vl' if speaker_id.startswith('V') else 'nl' # Simple assumption
            original_wav_path = os.path.join(jasmin_audio_base_dir, component, region_code, original_wav_filename)

            # --- Load JSON ---
            if not os.path.exists(json_filepath):
                logging.warning(f"JSON file not found for {original_wav_filename} at {json_filepath}. Skipping.")
                continue
            try:
                with open(json_filepath, 'r', encoding='utf-8') as f:
                    transcription_data = json.load(f)
            except Exception as e:
                logging.warning(f"Error loading JSON {json_filepath}: {e}. Skipping.")
                continue

            # --- Load Original Audio ---
            if not os.path.exists(original_wav_path):
                logging.warning(f"Original WAV file not found for {original_wav_filename} at {original_wav_path}. Skipping.")
                continue
            try:
                full_audio = AudioSegment.from_wav(original_wav_path)
            except Exception as e:
                logging.warning(f"Error loading WAV {original_wav_path}: {e}. Skipping.")
                continue

            # --- Process Segments ---
            if 'whisper_segments' not in transcription_data or not transcription_data['whisper_segments']:
                logging.warning(f"No segments found in JSON {json_filepath} (expected key 'whisper_segments'). Skipping.")
                continue

            for i, segment in enumerate(transcription_data['whisper_segments']):
                start_time = segment.get('start')
                end_time = segment.get('end')
                segment_text = segment.get('text', '').strip()

                if start_time is None or end_time is None or not segment_text:
                    continue

                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                min_duration_ms = 100 # Optional: Minimum duration check

                if start_ms >= end_ms or start_ms < 0 or end_ms > len(full_audio) or (end_ms - start_ms < min_duration_ms):
                    # logging.warning(f"Invalid/short segment times ({start_ms}-{end_ms} ms) for {file_root} segment {i}. Skipping.")
                    continue

                utt_id = f"{speaker_id}_{file_root}_seg{i:04d}"
                cleaned_text = clean_text(segment_text)
                if not cleaned_text:
                    continue

                try:
                    audio_segment = full_audio[start_ms:end_ms]
                except Exception as e:
                    logging.warning(f"Error slicing audio for {utt_id} ({start_ms}-{end_ms} ms): {e}. Skipping.")
                    continue

                segment_wav_filename = f"{utt_id}.wav"
                segment_wav_path = os.path.join(segmented_audio_dir, segment_wav_filename)
                segment_wav_path_abs = os.path.abspath(segment_wav_path)

                try:
                    audio_segment.export(segment_wav_path, format="wav")
                except Exception as e:
                    logging.warning(f"Error exporting audio segment {segment_wav_path}: {e}. Skipping.")
                    continue

                wav_scp_list.append(f"{utt_id} {segment_wav_path_abs}")
                text_list.append(f"{utt_id} {cleaned_text}")
                utt2spk_list.append(f"{utt_id} {speaker_id}")

        except Exception as row_error:
            logging.error(f"Unexpected error processing row {index} for group {group_name}: {row_error}. Skipping row.")
            continue # Skip to next row in CSV

    # --- Write ESPnet Files for the current group ---
    if not wav_scp_list:
        logging.warning(f"No segments processed for group '{group_name}'. ESPnet files will be empty.")
        continue # Skip writing files if nothing was processed

    wav_scp_file = os.path.join(espnet_data_dir, 'wav.scp')
    text_file = os.path.join(espnet_data_dir, 'text')
    utt2spk_file = os.path.join(espnet_data_dir, 'utt2spk')
    spk2utt_file = os.path.join(espnet_data_dir, 'spk2utt')

    try:
        with open(wav_scp_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(wav_scp_list)) + '\n')

        with open(text_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(text_list)) + '\n')

        with open(utt2spk_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(utt2spk_list)) + '\n')

        logging.info(f"Successfully generated ESPnet files in {espnet_data_dir}:")
        logging.info(f"- {os.path.basename(wav_scp_file)} ({len(wav_scp_list)} entries)")
        logging.info(f"- {os.path.basename(text_file)} ({len(text_list)} entries)")
        logging.info(f"- {os.path.basename(utt2spk_file)} ({len(utt2spk_list)} entries)")

        # Generate spk2utt
        spk2utt_dict = {}
        for line in utt2spk_list:
            utt_id, spk_id = line.split(maxsplit=1)
            if spk_id not in spk2utt_dict:
                spk2utt_dict[spk_id] = []
            spk2utt_dict[spk_id].append(utt_id)

        with open(spk2utt_file, 'w', encoding='utf-8') as f:
            for spk_id in sorted(spk2utt_dict.keys()):
                utts = " ".join(sorted(spk2utt_dict[spk_id]))
                f.write(f"{spk_id} {utts}\n")
        logging.info(f"- {os.path.basename(spk2utt_file)} ({len(spk2utt_dict)} speakers)")

    except Exception as e:
        logging.error(f"Error writing ESPnet files for group '{group_name}': {e}")

logging.info("--- Data preparation finished for all groups ---")
