# --- Environment Variable Setup (Set these BEFORE importing PyTorch) ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Or your desired GPU index
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments, # Explicitly import for clarity
    GenerationConfig
)
import evaluate # Hugging Face Evaluate library
import sys # For exiting if no data
import time # Import time module for timing
import numpy as np # For manual evaluation
from torch.utils.data import DataLoader # For manual evaluation
import json # For saving eval_results.json

# --- Initial PyTorch and CUDA Check ---
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs available to PyTorch: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"--- GPU {i} ---")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GiB")
        # Initial allocated/reserved memory is usually 0 before operations
        print(f"  Allocated Memory (Initial): {torch.cuda.memory_allocated(i) / (1024**3):.2f} GiB")
        print(f"  Reserved Memory (Cached Initial): {torch.cuda.memory_reserved(i) / (1024**3):.2f} GiB")
else:
    print("CUDA is not available to PyTorch. Please check your installation.")
    # Consider exiting if GPU is essential
    # sys.exit("CUDA not available. Exiting.")


# --- 0. Configuration ---
BASE_MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "nl" # Primary language for tokenizer setup
TASK = "transcribe"
OUTPUT_DIR = "./whisper-finetuned-synthetic-only-large-v3"

SYNTHETIC_AUDIO_BASE_DIR = Path('./synthetic_audio_coqui')
NL_SYNTHETIC_METADATA_CSV = SYNTHETIC_AUDIO_BASE_DIR / 'nl' / 'nl_metadata.csv'
FR_SYNTHETIC_METADATA_CSV = SYNTHETIC_AUDIO_BASE_DIR / 'fr' / 'fr_metadata.csv'

# Training Hyperparameters
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4 # Used for manual evaluation DataLoader
NUM_TRAIN_EPOCHS = 5 # We'll train for 1 epoch and then evaluate
LEARNING_RATE = 1e-5
WARMUP_STEPS = 50
GRADIENT_ACCUMULATION_STEPS = 1
FP16 = torch.cuda.is_available()
GENERATION_MAX_LENGTH = 225 # Max length for generated sequences
MAX_SYNTHETIC_EVAL_SAMPLES = 1000 # Max samples for synthetic validation to save memory

# --- 1. Load Synthetic Data ---
print("\n--- Loading Synthetic Data ---")

def load_synthetic_data_from_metadata(metadata_csv_path: Path, audio_subdir_name: str, base_audio_dir: Path, lang_name: str):
    data = []
    if metadata_csv_path.exists():
        df = pd.read_csv(metadata_csv_path)
        print(f"Found {len(df)} entries in {lang_name} synthetic metadata ({metadata_csv_path.name}).")
        for _, row in df.iterrows():
            audio_filename = row.get('output_filename')
            original_sentence = row.get('original_sentence')

            if audio_filename is None or original_sentence is None:
                print(f"Warning: Missing 'output_filename' or 'original_sentence' in {metadata_csv_path.name}. Skipping row.")
                continue

            audio_path = base_audio_dir / audio_subdir_name / str(audio_filename)
            if audio_path.exists() and audio_path.stat().st_size > 0:
                data.append({'audio': str(audio_path), 'sentence': str(original_sentence)})
            else:
                print(f"Warning: {lang_name} synthetic audio file not found or empty: {audio_path}")
    else:
        print(f"Warning: {lang_name} synthetic metadata file not found: {metadata_csv_path}")
    return data

all_synthetic_data = []
all_synthetic_data.extend(load_synthetic_data_from_metadata(NL_SYNTHETIC_METADATA_CSV, 'nl', SYNTHETIC_AUDIO_BASE_DIR, "NL"))
all_synthetic_data.extend(load_synthetic_data_from_metadata(FR_SYNTHETIC_METADATA_CSV, 'fr', SYNTHETIC_AUDIO_BASE_DIR, "FR"))

if not all_synthetic_data:
    print("No synthetic data loaded. Please check paths. Exiting.")
    sys.exit(1)

print(f"Total synthetic samples loaded: {len(all_synthetic_data)}")
raw_dataset = Dataset.from_pandas(pd.DataFrame(all_synthetic_data))
raw_dataset = raw_dataset.filter(
    lambda x: x["sentence"] is not None and isinstance(x["sentence"], str) and len(x["sentence"].strip()) > 1
)
print(f"Total synthetic samples after filtering: {len(raw_dataset)}")

if len(raw_dataset) == 0:
    print("Dataset is empty after filtering. Exiting.")
    sys.exit(1)

if len(raw_dataset) < 10:
    print("Synthetic dataset too small to split. Using all for training and evaluation (not ideal).")
    dataset_dict = DatasetDict({'train': raw_dataset, 'test': raw_dataset})
else:
    train_test_split = raw_dataset.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=16000))
print("\nSynthetic dataset prepared and split:")
print(dataset_dict)

# --- 2. Load Processor (Feature Extractor and Tokenizer) ---
print(f"\n--- Loading Processor for {BASE_MODEL_NAME} ---")
try:
    feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL_NAME, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME, language=LANGUAGE, task=TASK)
except Exception as e:
    print(f"Error loading processor for {LANGUAGE}. Trying without language/task specific args: {e}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL_NAME)
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# --- 3. Prepare Data for Model (Preprocessing) ---
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("\nPreprocessing synthetic dataset...")
tokenized_datasets = dataset_dict.map(
    prepare_dataset,
    remove_columns=dataset_dict.column_names["train"],
    num_proc=1 # Using 1 process for stability; increase if safe and dataset is large
)
print("Synthetic dataset preprocessed.")
print(tokenized_datasets)

# --- 4. Define Data Collator ---
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- 5. Define Evaluation Metric (WER) ---
wer_metric = evaluate.load("wer")

# --- 6. Load Pre-trained Model ---
print(f"\n--- Loading Pre-trained Model: {BASE_MODEL_NAME} ---")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# --- 7. Define Training Arguments ---
print("\n--- Defining Training Arguments ---")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    fp16=FP16,
    gradient_checkpointing=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=25,
    eval_strategy="no",
    save_strategy="no",
    dataloader_num_workers=0,
    optim="adamw_torch",
)

# --- 8. Initialize Trainer ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    # tokenizer=processor.feature_extractor
    tokenizer=processor
)

# --- 9. Start Fine-Tuning ---
print("\n--- Starting Fine-Tuning on Synthetic Data Only ---")
start_time = time.time()

try:
    train_result = trainer.train()
    end_time = time.time()
    training_duration_seconds = end_time - start_time
    print(f"Fine-tuning finished in {training_duration_seconds:.2f} seconds ({(training_duration_seconds/3600):.2f} hours).")

    # --- 10. Save Model and Metrics ---
    print(f"\nSaving model trained for {NUM_TRAIN_EPOCHS} epoch(s) to {OUTPUT_DIR}")
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_duration_seconds"] = training_duration_seconds
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    print(f"Training metrics: {metrics}")

    # --- 11. Manually Evaluate Final Model on the (Synthetic) Test Set ---
    print("\n--- Manually Evaluating the Model on the Synthetic Test Set ---")
    
    # --- Optional: Reduce the size of the synthetic test set for this evaluation ---
    if len(tokenized_datasets["test"]) > MAX_SYNTHETIC_EVAL_SAMPLES:
        print(f"INFO: Using a subset of {MAX_SYNTHETIC_EVAL_SAMPLES} samples for synthetic evaluation (original size: {len(tokenized_datasets['test'])}).")
        synthetic_eval_dataset_for_manual_eval = tokenized_datasets["test"].shuffle(seed=42).select(range(MAX_SYNTHETIC_EVAL_SAMPLES))
    else:
        synthetic_eval_dataset_for_manual_eval = tokenized_datasets["test"]

    device = model.device
    model.eval() 

    eval_dataloader = DataLoader(
        synthetic_eval_dataset_for_manual_eval,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=data_collator,
        num_workers=0
    )

    all_predictions_ids = []
    all_labels_ids = []

    generation_config = GenerationConfig(
        max_length=GENERATION_MAX_LENGTH,
    )

    print(f"Starting manual evaluation on {device} with batch size {EVAL_BATCH_SIZE}...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            print(f"  Processing batch {batch_idx + 1}/{len(eval_dataloader)}")
            input_features = batch["input_features"].to(device)
            labels = batch["labels"] 

            generated_ids = model.generate(
                inputs=input_features,
                generation_config=generation_config
            )
            
            all_predictions_ids.extend(generated_ids.cpu().numpy())
            all_labels_ids.extend(labels.numpy()) 
    
    print("Manual evaluation: Decoding predictions and labels...")
    processed_labels_ids = [
        np.where(label_seq == -100, processor.tokenizer.pad_token_id, label_seq)
        for label_seq in all_labels_ids
    ]

    pred_str = processor.batch_decode(all_predictions_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(processed_labels_ids, skip_special_tokens=True)

    print("Manual evaluation: Computing WER...")
    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)
    
    eval_metrics_to_save = {
        f"eval_wer": wer_score,
        f"eval_samples": len(synthetic_eval_dataset_for_manual_eval),
    }
    trainer.log_metrics("eval", eval_metrics_to_save) 
    
    eval_results_path = Path(OUTPUT_DIR) / "eval_results.json"
    with open(eval_results_path, "w") as f:
        json.dump(eval_metrics_to_save, f, indent=4)
    
    print(f"Manual evaluation results on synthetic test set: {eval_metrics_to_save}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\nCUDA cache cleared.")

print(f"\n--- Script complete. Model saved to {OUTPUT_DIR} ---")
print("You can now load your fine-tuned model using:")
print(f"  from transformers import WhisperForConditionalGeneration, WhisperProcessor")
print(f"  model = WhisperForConditionalGeneration.from_pretrained('{OUTPUT_DIR}')")
print(f"  processor = WhisperProcessor.from_pretrained('{OUTPUT_DIR}')")
print("\nIMPORTANT: Now, evaluate this model on your REAL, HELD-OUT JASMIN test sets (NL and FR)")
print("to assess its performance on actual speech and analyze phoneme-level changes.")
