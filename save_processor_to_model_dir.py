from transformers import WhisperProcessor
import os
import sys

# --- Configuration ---
# IMPORTANT: Set this to the correct path of your fine-tuned model directory
MODEL_OUTPUT_DIR = "./whisper-finetuned-synthetic-only-turbo"  # Or the full absolute path
# MODEL_OUTPUT_DIR = "/full/path/to/your/whisper-finetuned-synthetic-only-turbo" # Example absolute path

# This MUST be the base model name that was used when fine-tuning the model
# in MODEL_OUTPUT_DIR. Your fine-tuning script used "openai/whisper-large".
# If the "turbo" model was based on a different Whisper variant (e.g., medium, small),
# you MUST change this value accordingly.
BASE_MODEL_NAME = "openai/whisper-large-v3-turbo"

# These should match the settings used during the fine-tuning of the "turbo" model
LANGUAGE = "nl"
TASK = "transcribe"

# --- Get Absolute Path for Clarity and Robustness ---
absolute_model_output_dir = os.path.abspath(MODEL_OUTPUT_DIR)

if not os.path.isdir(absolute_model_output_dir):
    print(f"Error: Model directory not found or is not a directory at: {absolute_model_output_dir}")
    print("Please ensure the MODEL_OUTPUT_DIR path is correct.")
    sys.exit(1)
else:
    print(f"Target model directory: {absolute_model_output_dir}")
    print(f"Attempting to save processor (from base: {BASE_MODEL_NAME}, lang: {LANGUAGE}, task: {TASK}) to this directory.")
    try:
        # Load the processor configured as it would have been during fine-tuning
        print(f"Loading processor from Hugging Face Hub: {BASE_MODEL_NAME}")
        processor = WhisperProcessor.from_pretrained(
            BASE_MODEL_NAME,
            language=LANGUAGE,
            task=TASK
        )

        # Save this processor to your fine-tuned model's directory
        processor.save_pretrained(absolute_model_output_dir)
        print(f"Processor (including tokenizer and feature extractor) successfully saved to {absolute_model_output_dir}.")
        print("The directory should now contain the necessary tokenizer and feature extractor files, such as:")
        print("- tokenizer.json")
        print("- vocab.json")
        print("- merges.txt")
        print("- tokenizer_config.json")
        print("- special_tokens_map.json")
        print("- preprocessor_config.json")
        print("\nFiles currently in the directory:")
        for item in sorted(os.listdir(absolute_model_output_dir)): # sorted for consistent listing
            print(f"- {item}")
        print("\nYou should now be able to load this model with the Hugging Face pipeline or from_pretrained methods.")

    except Exception as e:
        print(f"An error occurred while loading or saving the processor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
