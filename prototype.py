import yt_dlp
import whisper
import os
import torch # For checking CUDA and for Hugging Face pipeline
from transformers import pipeline as hf_pipeline # Alias to avoid confusion with whisper.pipeline
import json # For potential future detailed output, though not strictly used in this version's SRT

def format_time_for_srt(seconds: float) -> str:
    """Converts seconds to SRT time format HH:MM:SS,mmm"""
    if seconds is None: # Should be filtered out before calling, but as a safeguard
        return "00:00:00,000"
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    mins = seconds // 60
    secs = seconds % 60
    hours = mins // 60
    mins = mins % 60
    return f"{hours:02}:{mins:02}:{secs:02},{millis:03}"

def generate_srt_file(transcription_result: dict, srt_filename: str):
    """
    Generates an SRT file from Whisper's transcription result (openai-whisper format)
    or a similarly structured dictionary from the Hugging Face pipeline.
    The transcription_result dictionary must have a "segments" key,
    which is a list of dictionaries, each with "start", "end", and "text".
    """
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        segment_num = 1
        for segment in transcription_result.get("segments", []):
            start_time_val = segment.get("start")
            end_time_val = segment.get("end")
            text = segment.get("text", "").strip()

            if start_time_val is None or end_time_val is None or not text:
                # Skip segments without proper timing or empty text for SRT
                # print(f"Skipping segment for SRT due to missing data: start={start_time_val}, end={end_time_val}, text='{text[:20]}...'")
                continue

            start_time_srt = format_time_for_srt(start_time_val)
            end_time_srt = format_time_for_srt(end_time_val)
            
            srt_file.write(f"{segment_num}\n")
            srt_file.write(f"{start_time_srt} --> {end_time_srt}\n")
            srt_file.write(f"{text}\n\n")
            segment_num += 1
    if segment_num > 1:
        print(f"SRT file saved to {srt_filename}")
    else:
        print(f"SRT file {srt_filename} was not generated as no valid segments were found.")


def transcribe_audio_with_openai_whisper(
    audio_path: str,
    model_name: str,
    output_txt_filename: str,
    output_srt_filename: str,
    language: str = None # Specify language code e.g. "en", "nl", or None for auto-detect
):
    """Loads an openai-whisper model, transcribes audio, and saves .txt and .srt files."""
    print(f"\nLoading openai-whisper model: {model_name}...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_name, device=device)
        print(f"Transcribing with openai-whisper model '{model_name}' (this may take a while)...")
        
        transcribe_options = {"language": language} if language else {}
        # word_timestamps=True gives more detailed segment data, though not all used by this SRT func
        result = model.transcribe(audio_path, verbose=False, word_timestamps=True, **transcribe_options)

        with open(output_txt_filename, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"Openai-whisper transcription (text) saved to {output_txt_filename}")

        generate_srt_file(result, output_srt_filename)
        
    except Exception as e:
        print(f"Error during openai-whisper transcription with '{model_name}': {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure the model name is correct and you have enough resources (e.g., VRAM for larger models).")

def transcribe_audio_with_hf_pipeline(
    audio_path: str,
    model_path_or_id: str,
    output_txt_filename: str,
    output_srt_filename: str,
    language: str, # Language code is generally required for HF pipeline's generate_kwargs
    chunk_length_s: int = 30 # As used in your script
):
    """Loads a Hugging Face ASR pipeline, transcribes audio, and saves .txt and .srt files."""
    print(f"\nLoading Hugging Face ASR pipeline with model: {model_path_or_id}...")
    # For Hugging Face pipeline, device can be 0 for cuda:0, 1 for cuda:1, or -1 for CPU
    device_pipeline = 0 if torch.cuda.is_available() else -1 
    try:
        asr_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_path_or_id,
            device=device_pipeline
        )
        print(f"Transcribing with Hugging Face model '{model_path_or_id}' (this may take a while)...")
        
        # Using parameters similar to your script
        # `return_timestamps=True` or `return_timestamps="segments"` should yield 'chunks'
        pipeline_output = asr_pipe(
            audio_path,
            chunk_length_s=chunk_length_s,
            generate_kwargs={
                "language": language, # Your script specified this
                "task": "transcribe"  # Your script specified this
            },
            return_timestamps=True # To get 'chunks' with timestamps
        )

        full_text = pipeline_output["text"]
        
        # Prepare segments for SRT generation (matching the structure generate_srt_file expects)
        hf_segments_for_srt = []
        if "chunks" in pipeline_output:
            for chunk in pipeline_output["chunks"]:
                text = chunk.get("text", "").strip()
                timestamp = chunk.get("timestamp")

                if not text or not timestamp: # Skip if no text or no timestamp tuple
                    continue

                start_time, end_time = timestamp
                
                # Critical for SRT: ensure start and end times are valid floats
                if start_time is None or end_time is None:
                    print(f"Warning: Hugging Face segment with text '{text[:50]}...' has None timestamp ({start_time}, {end_time}). Skipping for SRT.")
                    continue
                
                hf_segments_for_srt.append({
                    "start": float(start_time),
                    "end": float(end_time),
                    "text": text
                })
        else:
            print("Warning: 'chunks' not found in Hugging Face pipeline output. SRT file may be empty or incomplete.")
            # Fallback: create one large segment if no chunks but text exists
            if full_text:
                 hf_segments_for_srt.append({"start": 0.0, "end": 30.0, "text": full_text}) # Arbitrary times
                 print("Fallback: Created a single segment for SRT from full text.")


        with open(output_txt_filename, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Hugging Face transcription (text) saved to {output_txt_filename}")

        # The generate_srt_file function expects a dict with a "segments" key
        srt_result_data = {"text": full_text, "segments": hf_segments_for_srt}
        generate_srt_file(srt_result_data, output_srt_filename)

    except Exception as e:
        print(f"Error during Hugging Face ASR pipeline transcription with '{model_path_or_id}': {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure the model path/ID is correct, the language is appropriate, and you have enough resources.")


def main():
    youtube_url = input("Enter the YouTube video URL: ")
    
    downloaded_audio_basename = "youtube_audio_for_transcription"
    downloaded_audio_wav = f"{downloaded_audio_basename}.wav"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{downloaded_audio_basename}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'keepvideo': False,
        'noplaylist': True,
        'quiet': False,
        'progress': True,
    }

    print(f"\nDownloading and converting audio from: {youtube_url}")
    try:
        # Ensure old audio file is removed if it exists, to prevent confusion
        if os.path.exists(downloaded_audio_wav):
            os.remove(downloaded_audio_wav)
            print(f"Removed existing file: {downloaded_audio_wav}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([youtube_url])
            if error_code != 0:
                print(f"yt-dlp failed with error code: {error_code}")
                return

        if not os.path.exists(downloaded_audio_wav):
            print(f"Error: Expected audio file '{downloaded_audio_wav}' not found after download.")
            print("Please check yt-dlp's output above for any errors during download or conversion.")
            return
        print(f"Audio successfully downloaded and saved as: {downloaded_audio_wav}")
    except Exception as e:
        print(f"An error occurred during audio download/conversion: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Configuration for Transcription ---
    print("\n--- Baseline Model (using openai-whisper) ---")
    baseline_model_name = input("Enter baseline Whisper model name (e.g., 'base', 'small', 'medium', 'large-v3'): ").strip()
    baseline_language = input("Enter language code for baseline model (e.g., 'en', 'nl', or leave blank for auto-detect): ").strip() or None
    
    print("\n--- Fine-tuned Model (using Hugging Face Transformers) ---")
    finetuned_model_path = input("Enter path/ID to your Hugging Face fine-tuned model (e.g., '/path/to/model' or 'username/model-name'): ").strip()
    finetuned_language = input(f"Enter language code for fine-tuned model (e.g., 'nl', 'en' - this is usually fixed for the model): ").strip()

    baseline_output_txt = "transcription_baseline.txt"
    baseline_output_srt = "transcription_baseline.srt"
    finetuned_output_txt = "transcription_finetuned_hf.txt"
    finetuned_output_srt = "transcription_finetuned_hf.srt"

    # --- 1. Transcribe with Baseline (openai-whisper) Model ---
    if baseline_model_name:
        transcribe_audio_with_openai_whisper(
            downloaded_audio_wav,
            baseline_model_name,
            baseline_output_txt,
            baseline_output_srt,
            language=baseline_language
        )
    else:
        print("\nNo baseline model name provided. Skipping baseline transcription.")

    # --- 2. Transcribe with Fine-tuned (Hugging Face) Model ---
    if not finetuned_model_path:
        print("\nNo path/ID provided for the fine-tuned Hugging Face model. Skipping fine-tuned transcription.")
    elif not finetuned_language:
        print("\nLanguage code for the fine-tuned Hugging Face model is required. Skipping fine-tuned transcription.")
    else:
        transcribe_audio_with_hf_pipeline(
            downloaded_audio_wav,
            finetuned_model_path,
            finetuned_output_txt,
            finetuned_output_srt,
            language=finetuned_language
        )

    print("\n--- Process Complete ---")
    print("Generated files (if successful):")
    if os.path.exists(baseline_output_txt): print(f"  - Baseline Transcription: {baseline_output_txt}")
    if os.path.exists(baseline_output_srt): print(f"  - Baseline Subtitles: {baseline_output_srt}")
    if os.path.exists(finetuned_output_txt): print(f"  - Fine-tuned (HF) Transcription: {finetuned_output_txt}")
    if os.path.exists(finetuned_output_srt): print(f"  - Fine-tuned (HF) Subtitles: {finetuned_output_srt}")
    
    print("\nYou can now open the .srt files in a video editor to overlay them on your video,")
    print("or compare the .txt files directly.")

if __name__ == "__main__":
    main()
