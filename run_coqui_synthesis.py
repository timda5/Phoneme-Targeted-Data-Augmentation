import os
import re
import sys
import time
import csv # Import the csv module
import random # Import the random module for sampling
from pathlib import Path # Import Path
from tqdm import tqdm
import soundfile as sf
from TTS.api import TTS

# --- Configuration ---

# 1. Paths
# Use Path object, '.' assumes script runs from the base dir containing 'jasmin-data', 'analysis_outputs' etc.
# Adjust base_path if your script is located elsewhere relative to these folders.
base_path = Path('.')
jasmin_base_path = base_path / 'jasmin-data' / 'Data' / 'data'
analysis_output_folder = base_path / 'analysis_outputs'
# Assuming sentence files might be in analysis_outputs or the base_path, adjust as needed
# cleaned_sentences_folder = base_path

id_files = {
    'nl': analysis_output_folder / 'lowest_error_utt_ids_nl_large3turbo.txt', # Example filename, adjust as needed
    'fr': analysis_output_folder / 'lowest_error_utt_ids_fr_large3turbo.txt', # Example filename, adjust as needed
}
sentence_files = {
    # Point these to your actual sentence files (e.g., the ones with one sentence per line)
    'nl': analysis_output_folder / 'output_nl_atleast_one_cleaned_turbo.txt', # Example filename, adjust as needed
    'fr': analysis_output_folder / 'output_fr_atleast_one_cleaned_turbo.txt', # Example filename, adjust as needed
}
synthetic_audio_base_dir = base_path / 'synthetic_audio_coqui/turbo'

# 2. Coqui TTS Settings
tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
target_language = "nl" # Set default, will be overridden in loop if needed
use_gpu = True

# 3. Cloning Settings
cloning_snippet_duration_sec = 15

# 4. --- Sentence Sampling ---
# Set the number of sentences to randomly sample per voice.
# Set to None to synthesize all sentences.
num_sentences_per_voice = 1500 # Example: Sample 1500 sentences

# --- Helper Functions ---
def get_wav_path(utt_id, jasmin_root):
    """Constructs the expected path to a JASMIN WAV file based on the utt_id."""
    match = re.match(r'([NV]\d+)_([fv]\w+)', utt_id)
    if not match:
        print(f"Warning: Could not parse file identifier from utterance ID: {utt_id}")
        return None
    speaker_id, file_id = match.groups()
    file_prefix = file_id[:2]
    # Determine subdirectory based on file prefix ('fv' or 'fn')
    if file_prefix == 'fv':
        sub_dir = 'vl' # Flemish speakers
    elif file_prefix == 'fn':
        sub_dir = 'nl' # Netherlands speakers (assuming based on common JASMIN structure)
        # If 'fn' should also go to 'vl', change 'nl' back to 'vl' here.
    else:
        print(f"Warning: Unknown file prefix '{file_prefix}' in ID: {utt_id}")
        return None

    wav_filename = f"{file_id}.wav"
    # Construct the full path using pathlib
    full_path = Path(jasmin_root) / 'audio' / 'wav' / 'comp-q' / sub_dir / wav_filename

    if not full_path.exists():
        # Attempt alternative path if the first guess fails (e.g., if fn/fv mapping is uncertain)
        alt_sub_dir = 'vl' if sub_dir == 'nl' else 'nl'
        alt_path = Path(jasmin_root) / 'audio' / 'wav' / 'comp-q' / alt_sub_dir / wav_filename
        if alt_path.exists():
            # print(f"Note: Found wav file using alternative path: {alt_path}") # Optional: uncomment for debugging
            return alt_path
        else:
            print(f"Warning: Wav file not found at expected path: {full_path} or alternative: {alt_path}")
            return None
    return full_path

def load_cloning_snippet(wav_path, duration_sec):
    """Loads the initial segment of a WAV file for voice cloning."""
    try:
        wav_path_obj = Path(wav_path) # Ensure it's a Path object
        info = sf.info(str(wav_path_obj)) # sf.info might need string path
        samplerate = info.samplerate
        frames_to_read = int(samplerate * duration_sec)
        # Ensure we read at least 1 second, even if duration_sec is less
        min_frames = int(samplerate * 1.0)
        frames_to_read = max(frames_to_read, min_frames)

        # Check total frames available
        if info.frames < min_frames:
             print(f"Warning: Source audio file {wav_path_obj} is too short ({info.duration:.2f}s) for cloning (min 1s required).")
             return None, None

        # Adjust frames_to_read if it exceeds file length
        frames_to_read = min(frames_to_read, info.frames)

        audio_data, sr = sf.read(str(wav_path_obj), frames=frames_to_read, dtype='float32', always_2d=False)

        if sr != samplerate:
            print(f"Warning: Sample rate mismatch for {wav_path_obj}. Expected {samplerate}, got {sr}.")
            # Depending on TTS requirements, you might want to resample here or return None

        actual_duration = len(audio_data) / samplerate
        if actual_duration < 1.0:
             print(f"Warning: Could not load sufficient audio ({actual_duration:.2f}s < 1.0s) from {wav_path_obj}")
             return None, None

        # Save snippet to a temporary file (Coqui often prefers file paths)
        # Use Path.name to get the filename part
        temp_snippet_path = f"temp_snippet_{wav_path_obj.name}"
        sf.write(temp_snippet_path, audio_data, samplerate)
        return temp_snippet_path, samplerate
    except Exception as e:
        print(f"Error loading audio snippet from {wav_path}: {e}")
        return None, None

# --- Main Synthesis Function (Modified for Sampling and Metadata) ---

def synthesize_audio(group_tag, id_filepath, sentence_filepath, output_group_dir, tts_instance, sample_size=None):
    """Synthesizes audio for a given group, optionally sampling sentences, and saves metadata."""
    print(f"\n--- Starting Synthesis for Group: {group_tag.upper()} ---")

    # Determine target language based on group tag
    current_target_language = group_tag # Assumes group_tag is 'nl' or 'fr'
    print(f"Target language set to: {current_target_language}")

    # 1. Read Utterance IDs
    id_filepath_obj = Path(id_filepath)
    if not id_filepath_obj.exists():
        print(f"Error: Utterance ID file not found: {id_filepath_obj}"); return
    with open(id_filepath_obj, 'r', encoding='utf-8') as f:
        utterance_ids = [line.strip() for line in f if line.strip()]
    if not utterance_ids: print(f"No utterance IDs found in {id_filepath_obj}"); return
    print(f"Found {len(utterance_ids)} utterance IDs (voices) for cloning.")

    # 2. Read ALL Sentences
    sentence_filepath_obj = Path(sentence_filepath)
    if not sentence_filepath_obj.exists():
        print(f"Error: Sentence file not found: {sentence_filepath_obj}"); return
    with open(sentence_filepath_obj, 'r', encoding='utf-8') as f:
        all_sentences = [line.strip() for line in f if line.strip()]
    if not all_sentences: print(f"No sentences found in {sentence_filepath_obj}"); return
    print(f"Found {len(all_sentences)} total sentences available.")

    # 3. Determine sentences to use (Sample or All)
    if sample_size is not None and sample_size < len(all_sentences):
        print(f"Sampling {sample_size} sentences per voice.")
        num_sentences_to_process = sample_size # We'll sample inside the loop
    else:
        print("Using all available sentences per voice.")
        num_sentences_to_process = len(all_sentences)

    # 4. Create output directory
    output_group_dir_obj = Path(output_group_dir)
    output_group_dir_obj.mkdir(parents=True, exist_ok=True) # Use Path.mkdir
    print(f"Output directory: {output_group_dir_obj}")

    # --- Metadata File Setup ---
    metadata_filepath = output_group_dir_obj / f"{group_tag}_metadata.csv"
    print(f"Metadata will be saved to: {metadata_filepath}")
    # Check if the file exists before starting the loop
    file_exists = metadata_filepath.exists()

    # 5. Loop through voices
    total_synthesized = 0
    total_skipped_exist = 0
    total_skipped_error = 0
    temp_snippet_files = []

    # Open metadata file once before the loop for efficiency
    try:
        with open(metadata_filepath, 'a', newline='', encoding='utf-8') as meta_f:
            writer = csv.writer(meta_f)
            # Write header only if the file is new or was empty before opening
            if not file_exists or metadata_filepath.stat().st_size == 0:
                writer.writerow(['output_filename', 'speaker_id', 'original_sentence'])
                # No need to track file_exists separately now, opening in 'a' handles it

            for utt_id in tqdm(utterance_ids, desc=f"Cloning Voices ({group_tag.upper()})", unit=" voice"):
                wav_path = get_wav_path(utt_id, jasmin_base_path)
                if not wav_path:
                    # Estimate errors based on planned sentences for this voice
                    total_skipped_error += num_sentences_to_process
                    print(f"\nWarning: Skipping voice {utt_id} - could not find WAV path.")
                    continue # Skip to the next voice

                snippet_path, snippet_sr = load_cloning_snippet(wav_path, cloning_snippet_duration_sec)
                if not snippet_path:
                    # Estimate errors based on planned sentences for this voice
                    total_skipped_error += num_sentences_to_process
                    print(f"\nWarning: Skipping voice {utt_id} - could not load cloning snippet.")
                    continue # Skip to the next voice
                temp_snippet_files.append(snippet_path) # Add for later cleanup

                # --- Select/Sample sentences FOR THIS VOICE ---
                if sample_size is not None and sample_size < len(all_sentences):
                    # Random sampling logic
                    sentences_for_this_voice = random.sample(all_sentences, sample_size)
                else:
                    # Use all sentences from the input file
                    sentences_for_this_voice = all_sentences # Use all

                # 6. Loop through the selected sentences for the current voice
                sentence_pbar = tqdm(enumerate(sentences_for_this_voice), total=len(sentences_for_this_voice), desc=f"  Synthesizing ({utt_id})", unit=" sentence", leave=False)
                for idx, sentence in sentence_pbar:
                    # Use a combination of utt_id and index for unique naming.
                    relative_output_filename = f"{utt_id}_synth_sample_{idx:04d}.wav"
                    full_output_filename = output_group_dir_obj / relative_output_filename

                    if full_output_filename.exists():
                        total_skipped_exist += 1
                        continue # Skip if file already exists

                    try:
                        # Basic sentence validation
                        if not sentence or not sentence.strip() or len(sentence.split()) < 1:
                            print(f"\nWarning: Skipping empty or invalid sentence for {utt_id}, index {idx}.")
                            total_skipped_error += 1
                            continue

                        # Perform TTS synthesis
                        tts_instance.tts_to_file(
                            text=sentence,
                            speaker_wav=snippet_path,
                            language=current_target_language, # Use language determined for the group
                            file_path=str(full_output_filename) # tts_to_file might prefer string path
                        )
                        total_synthesized += 1
                        # --- Write to metadata file ---
                        writer.writerow([relative_output_filename, utt_id, sentence])

                    except Exception as e:
                        print(f"\nWarning: Skipping sentence due to TTS error for {utt_id}, sentence idx {idx}.")
                        # print(f"  Error: {e}") # Uncomment for detailed error diagnosis
                        # print(f"  Sentence: '{sentence[:100]}...'")
                        total_skipped_error += 1
                        # Optional: Add a small delay if errors are frequent and might be rate-related
                        # time.sleep(0.5)

    except IOError as e:
         print(f"\nFatal Error: Could not open or write to metadata file {metadata_filepath}: {e}")
         # If we can't open the metadata file at all, it's probably best to stop processing this group.
         return # Stop processing this group if metadata can't be written

    # Clean up temporary snippet files
    print("\nCleaning up temporary snippet files...")
    for temp_file in temp_snippet_files:
        try:
            temp_file_path = Path(temp_file)
            if temp_file_path.exists():
                os.remove(temp_file_path) # os.remove works with Path objects too
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")

    print(f"\n--- Finished Synthesis for Group: {group_tag.upper()} ---")
    print(f"Successfully synthesized: {total_synthesized}")
    print(f"Skipped (already exist): {total_skipped_exist}")
    print(f"Skipped (error/invalid): {total_skipped_error}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing Coqui TTS...")
    try:
        # Initialize TTS model
        tts = TTS(model_name=tts_model_name, progress_bar=True, gpu=use_gpu)
        print("TTS model loaded successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not load TTS model '{tts_model_name}'.")
        print(f"Error details: {e}")
        sys.exit(1) # Exit if TTS model fails to load

    # Process each language group defined in id_files/sentence_files
    for group in id_files.keys():
        if group not in sentence_files:
            print(f"Warning: Skipping group '{group}' - missing sentence file configuration.")
            continue

        output_dir = synthetic_audio_base_dir / group
        synthesize_audio(
            group_tag=group,
            id_filepath=id_files[group],
            sentence_filepath=sentence_files[group],
            output_group_dir=output_dir,
            tts_instance=tts,
            sample_size=num_sentences_per_voice # Pass the sample size here
        )

    print("\n\nAll processing finished.")
