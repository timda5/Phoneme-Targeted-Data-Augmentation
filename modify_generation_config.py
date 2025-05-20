import json
import os
import shutil 

# --- Configuration ---
json_file_path = "/home/tdamen/whisper-finetuned-synthetic-only-large-v3/generation_config.json"
key_to_modify = "forced_decoder_ids"
new_value = None 

# --- Create a backup ---
backup_file_path = json_file_path + ".bak"

try:
    if not os.path.exists(json_file_path):
        print(f"Error: Original file not found at {json_file_path}")
        exit(1)

    shutil.copy2(json_file_path, backup_file_path)
    print(f"Backup of original file created at: {backup_file_path}")

except Exception as e:
    print(f"Error creating backup: {e}")
    exit(1)

# --- Load, modify, and save the JSON file ---
try:
    # Read the original JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Successfully loaded JSON data from: {json_file_path}")

    # Modify the specific key
    if key_to_modify in data:
        original_value = data[key_to_modify]
        data[key_to_modify] = new_value
        print(f"Modified key '{key_to_modify}'. Original value: {original_value}, New value: {new_value}")
    else:
        print(f"Warning: Key '{key_to_modify}' not found in the JSON data. Adding it with the new value.")
        data[key_to_modify] = new_value


    # Write the modified data back to the original file path
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False) # indent=2 for pretty printing
    print(f"Successfully saved modified JSON data to: {json_file_path}")

    print("\n--- Verification ---")
    print(f"Content of the modified '{key_to_modify}' in {json_file_path}:")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        updated_data_check = json.load(f)
        print(json.dumps({key_to_modify: updated_data_check.get(key_to_modify)}, indent=2))


except json.JSONDecodeError as e:
    print(f"Error decoding JSON from {json_file_path}: {e}")
    print("Please ensure the file contains valid JSON.")
    print(f"The backup at {backup_file_path} remains untouched.")
except Exception as e:
    print(f"An error occurred during JSON processing: {e}")
    print(f"The backup at {backup_file_path} might be the last good state.")


# ```
# What forced_decoder_ids Does for Whisper

# For multilingual models like Whisper, forced_decoder_ids are crucial. They are a sequence of token IDs that you "force" the model to begin its generation with. Typically, for Whisper, this includes:

# A language token (e.g., <|nl|> for Dutch, <|en|> for English).
# A task token (e.g., <|transcribe|> or <|translate|>).
# Optionally, a token like <|notimestamps|> if you don't want timestamp predictions.
# By providing these, you guide the model to generate output in the correct language and perform the correct task. Your fine-tuning process also used these (implicitly via the language and task settings for the processor) to teach the model to expect these prompts.

# Why We Modified It in generation_config.json

# The ValueError you were encountering happened because:

# Your saved generation_config.json (from the fine-tuning process) had forced_decoder_ids set (e.g., for Dutch and transcribe).
# The Hugging Face pipeline for "automatic-speech-recognition", when you provide language="nl" and task="transcribe" (either directly in generate_kwargs or because the processor it loaded was configured that way), also tries to construct and set these forced_decoder_ids.
# This created a conflict: the model.generate() method was receiving these instructions from two sources (the model's static config and the pipeline's dynamic arguments), leading to the error.

# Is It Bad That We Changed It to null in generation_config.json?

# For your specific use case with the Hugging Face pipeline, no, it's not a bad thing; it was the correct solution to the conflict. Here's why:

# The Pipeline Takes Over: When you use the asr_pipeline and specify (or it infers from the processor) the language and task, the pipeline is designed to correctly set the forced_decoder_ids for you. By setting forced_decoder_ids: null in your generation_config.json, you remove the conflicting static configuration, allowing the pipeline to do its job without interference. The guiding tokens are still being provided, but now by the pipeline dynamically.

# Clarity and Control: This approach gives you clear control at the point of inference (your pipeline call). If you wanted to use the same fine-tuned model for a different task or language (if it were capable), you could change the pipeline's language and task arguments, and it would adjust the forced_decoder_ids accordingly.

# When Would It Be Bad?

# It would be problematic if:

# You were loading the model directly using WhisperForConditionalGeneration.from_pretrained(...) and then calling model.generate(...) without the pipeline and without manually providing the correct forced_decoder_ids (or equivalent arguments like decoder_input_ids that start with the language/task tokens) in your generate() call. In that scenario, the model, lacking these guiding tokens from its config, might default to English or try to auto-detect the language, which might not be what you want for your Dutch fine-tuned model.
# In your current setup (using the pipeline): You are correctly telling the pipeline what language and task to use. The pipeline then translates "language: nl, task: transcribe" into the appropriate forced_decoder_ids for the Whisper model. Modifying the generation_config.json to remove the static forced_decoder_ids simply resolves the conflict and lets the pipeline manage this aspect, which is its intended role.```