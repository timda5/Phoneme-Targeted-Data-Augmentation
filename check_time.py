import time
import os
import torch # For checking GPU availability
import whisper # Import here to be available for the transcribe_and_time function

# --- Configuration ---
AUDIO_FILE_PATH = "youtube_audio_for_transcription.wav" # Ensure this audio file exists
LANGUAGE = "nl" # Set the language for transcription, e.g., "en", "nl"
TARGET_GPU_ID = 1 # Specify the GPU ID you want to use (e.g., 0, 1). Set to None to let PyTorch decide.

print(f"--- PyTorch CUDA Diagnostics ---")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    try:
        current_device_id = torch.cuda.current_device()
        print(f"Current CUDA Device ID (before any explicit setting by script): {current_device_id}")
        print(f"Current CUDA Device Name (before any explicit setting by script): {torch.cuda.get_device_name(current_device_id)}")
        if TARGET_GPU_ID is not None:
            if TARGET_GPU_ID < torch.cuda.device_count():
                # Note: whisper.load_model(device=f"cuda:{TARGET_GPU_ID}") handles selection.
                # torch.cuda.set_device(TARGET_GPU_ID) could also be used here if preferred,
                # but the device string in load_model is usually sufficient.
                print(f"Script will attempt to use TARGET_GPU_ID: {TARGET_GPU_ID} ({torch.cuda.get_device_name(TARGET_GPU_ID)})")
            else:
                print(f"Warning: TARGET_GPU_ID {TARGET_GPU_ID} is out of range for available GPUs ({torch.cuda.device_count()}). Will use default GPU or CPU.")
        else:
            print(f"TARGET_GPU_ID is None. PyTorch will use its default GPU (usually cuda:0 if available).")
    except Exception as e:
        print(f"Error getting current CUDA device info: {e}")
print(f"---------------------------------")


# --- Helper Function to Transcribe and Time ---
def transcribe_and_time(model_name_or_path, audio_path, language, model_description):
    """
    Loads an openai-whisper model, transcribes audio, and returns the processing time, transcription, and the loaded model.
    The model is returned so it can be explicitly deleted outside this function.
    """
    print(f"\n--- Testing: {model_description} ({model_name_or_path}) ---")
    
    processing_time = -1
    transcription_text = "Transcription failed or was interrupted."
    loaded_model = None # Initialize to None
    device_used = "cpu" # Default to CPU


    try:
        if torch.cuda.is_available():
            if TARGET_GPU_ID is not None and TARGET_GPU_ID < torch.cuda.device_count():
                device_used = f"cuda:{TARGET_GPU_ID}"
            elif TARGET_GPU_ID is not None: # Target ID is invalid (out of range)
                 print(f"Warning: Invalid TARGET_GPU_ID {TARGET_GPU_ID}. Defaulting to cuda:0 or CPU if no GPUs.")
                 device_used = "cuda" # Let PyTorch pick default GPU or error out if none
            else: # TARGET_GPU_ID is None
                device_used = "cuda" # Let PyTorch pick default GPU (usually cuda:0)
        else:
            device_used = "cpu"
            
        print(f"Attempting to use device: {device_used}")
        
        if device_used.startswith("cuda"):
            torch.cuda.empty_cache() # Clear cache before loading

        print(f"Loading openai-whisper model: {model_name_or_path} to device '{device_used}'...")
        loaded_model = whisper.load_model(model_name_or_path, device=device_used)
        print("openai-whisper model loaded.")

        # Verify which device the model parameters are actually on
        if hasattr(loaded_model, 'device'):
            print(f"Model parameters are on device: {loaded_model.device}")
        else: # For older whisper versions or different model structures
            try:
                # A common way to check is the device of the first parameter
                param_device = next(loaded_model.parameters()).device
                print(f"Model (first parameter) is on device: {param_device}")
            except Exception:
                print("Could not determine model's parameter device directly.")


        start_time = time.time()
        result = loaded_model.transcribe(audio_path, language=language, verbose=False)
        end_time = time.time()
        transcription_text = result["text"]
        processing_time = end_time - start_time
        print(f"Transcription complete for {model_description}.")

    except Exception as e:
        print(f"Error with {model_description} ({model_name_or_path}): {e}")
        import traceback
        traceback.print_exc()
        
    return processing_time, transcription_text, loaded_model, device_used

# --- Main Comparison Logic ---
def main():
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"Error: Audio file not found at '{AUDIO_FILE_PATH}'.")
        print("Please make sure the audio file exists or update the AUDIO_FILE_PATH variable.")
        return

    print(f"\nStarting transcription speed comparison for audio file: {AUDIO_FILE_PATH}")
    print(f"Transcription Language: {LANGUAGE}")
    print("INFO: If the script is 'Killed', it's likely due to an Out-of-Memory error.")
    print("This version attempts to load and unload models sequentially to minimize memory overlap.")

    # --- Model 1: Baseline Model ---
    model_1_name = "large-v3"
    # model_1_name = "medium" # Try if large-v3 fails
    
    model_1_instance = None
    device_1 = "cpu" # Default
    time_1 = -1      # Default
    try:
        time_1, text_1, model_1_instance, device_1 = transcribe_and_time(
            model_1_name,
            AUDIO_FILE_PATH,
            LANGUAGE,
            f"Baseline Model ({model_1_name})"
        )
    finally:
        if model_1_instance is not None:
            print(f"Unloading Baseline Model ({model_1_name})...")
            del model_1_instance 
        if device_1.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"Baseline Model ({model_1_name}) processing finished and resources potentially freed.")


    # --- Model 2: Turbo Model ---
    model_2_name_or_path = "medium"
    # model_2_name_or_path = "/home/tdamen/whisper-finetuned-synthetic-only-large-v3" # Your fine-tuned model
    
    model_2_instance = None
    device_2 = "cpu" # Default
    time_2 = -1      # Default
    try:
        time_2, text_2, model_2_instance, device_2 = transcribe_and_time(
            model_2_name_or_path,
            AUDIO_FILE_PATH,
            LANGUAGE,
            f"Turbo Model ({os.path.basename(model_2_name_or_path)})"
        )
    finally:
        if model_2_instance is not None:
            print(f"Unloading Turbo Model ({os.path.basename(model_2_name_or_path)})...")
            del model_2_instance
        if device_2.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"Turbo Model ({os.path.basename(model_2_name_or_path)}) processing finished and resources potentially freed.")


    # --- Summary ---
    print("\n\n--- Speed Comparison Summary ---")

    if time_1 > 0:
        print(f"Time taken by Baseline Model ({model_1_name}) on device {device_1}: {time_1:.2f} seconds")
    else:
        print(f"Baseline Model ({model_1_name}) test did not complete successfully or was interrupted (used device: {device_1}).")

    if time_2 > 0:
        model_2_desc_print = f"Turbo Model ({os.path.basename(model_2_name_or_path)})"
        print(f"Time taken by {model_2_desc_print} on device {device_2}: {time_2:.2f} seconds")
        if time_1 > 0: 
            speed_up_factor = time_1 / time_2
            print(f"The {model_2_desc_print} is approximately {speed_up_factor:.2f} times faster than the Baseline Model ({model_1_name}) for this audio file.")
    else:
        print(f"Turbo Model ({os.path.basename(model_2_name_or_path)}) test did not complete successfully or was interrupted (used device: {device_2}).")

    print("\nNote: Ensure comparisons are run on the same hardware and under similar system load for fairness.")

if __name__ == "__main__":
    main()
