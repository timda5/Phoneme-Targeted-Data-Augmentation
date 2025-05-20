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
    GenerationConfig,
    EarlyStoppingCallback # Import EarlyStoppingCallback
)
import evaluate # Hugging Face Evaluate library
import sys # For exiting if no data
import time # Import time module for timing
import numpy as np # For manual evaluation
from torch.utils.data import DataLoader # For manual evaluation
import json # For saving eval_results.json
from transformers.trainer_utils import EvalPrediction # For compute_metrics

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
        print(f"  Allocated Memory (Initial): {torch.cuda.memory_allocated(i) / (1024**3):.2f} GiB")
        print(f"  Reserved Memory (Cached Initial): {torch.cuda.memory_reserved(i) / (1024**3):.2f} GiB")
else:
    print("CUDA is not available to PyTorch. Please check your installation.")
    # sys.exit("CUDA not available. Exiting.")


# --- 0. Configuration ---
BASE_MODEL_NAME = "openai/whisper-large-v3"
# LANGUAGE and TASK are useful for initializing the tokenizer's default state.
# For generation with mixed data, we won't force these on model.config.
LANGUAGE = "nl" # Primary/default language if needed for tokenizer
TASK = "transcribe"
OUTPUT_DIR = "./whisper-finetuned-large-v3-20epochs_mixed_data_no_force"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

SYNTHETIC_AUDIO_BASE_DIR = Path('./synthetic_audio_coqui') # Ensure this path is correct
NL_SYNTHETIC_METADATA_CSV = SYNTHETIC_AUDIO_BASE_DIR / 'nl' / 'nl_metadata.csv'
FR_SYNTHETIC_METADATA_CSV = SYNTHETIC_AUDIO_BASE_DIR / 'fr' / 'fr_metadata.csv'

# Training Hyperparameters
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE_TRAINER = 16    # per_device_eval_batch_size for trainer
EVAL_BATCH_SIZE_MANUAL = 4      # For your manual evaluation DataLoader at the end
NUM_TRAIN_EPOCHS = 20           # Target number of epochs
LEARNING_RATE = 1e-5
LR_SCHEDULER_TYPE = "constant_with_warmup"
WARMUP_STEPS = 50
GRADIENT_ACCUMULATION_STEPS = 1
FP16 = torch.cuda.is_available()
FP16_FULL_EVAL = True
GENERATION_MAX_LENGTH = 225
MAX_SYNTHETIC_EVAL_SAMPLES = 500 # Capping evaluation samples for trainer's eval
EARLY_STOPPING_PATIENCE = 3     # Patience for early stopping (epochs)
EARLY_STOPPING_THRESHOLD = 0.0  # Threshold for improvement (0.0 means any improvement counts)
SAVE_TOTAL_LIMIT = EARLY_STOPPING_PATIENCE + 2 # Keep a few recent checkpoints + the best one

# --- 1. Load Synthetic Data ---
print("\n--- Loading Synthetic Data (NL and FR) ---")
# We assume FR data has FRENCH sentences for French audio, and NL data has DUTCH sentences for Dutch audio.
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

# Splitting logic from your "new code"
if len(raw_dataset) < 10:
    print("Synthetic dataset very small. Using all for training and a small part for evaluation.")
    test_split_size = min(len(raw_dataset) -1 if len(raw_dataset) > 1 else 0, max(1, int(len(raw_dataset) * 0.1)))
    if test_split_size == 0 and len(raw_dataset) > 1: test_split_size = 1
    
    if test_split_size > 0 :
        train_indices = list(range(len(raw_dataset)))
        test_indices = train_indices[-test_split_size:]
        train_indices = train_indices[:-test_split_size]
        if not train_indices :
            train_indices = test_indices
            print("WARN: Train set was empty after split, using test set for train too.")
        dataset_dict = DatasetDict({'train': raw_dataset.select(train_indices), 'test': raw_dataset.select(test_indices)})
    else:
        dataset_dict = DatasetDict({'train': raw_dataset, 'test': raw_dataset})
else:
    train_test_split = raw_dataset.train_test_split(test_size=0.1, seed=42)
    if len(train_test_split['test']) == 0 and len(train_test_split['train']) > 0:
        print("Warning: Test split resulted in 0 samples. Taking one from train for evaluation.")
        dataset_dict = DatasetDict({
            'train': train_test_split['train'].select(range(len(train_test_split['train'])-1)),
            'test': train_test_split['train'].select([len(train_test_split['train'])-1])
        })
    else:
        dataset_dict = DatasetDict({'train': train_test_split['train'], 'test': train_test_split['test']})

dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=16000))
print("\nSynthetic dataset prepared and split:")
print(dataset_dict)


# --- 2. Load Processor (Feature Extractor and Tokenizer) ---
print(f"\n--- Loading Processor for {BASE_MODEL_NAME} ---")
# Initialize with LANGUAGE and TASK. This sets the tokenizer's default .language and .task attributes.
try:
    feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL_NAME, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME, language=LANGUAGE, task=TASK)
    print(f"Processor and Tokenizer initialized with language='{tokenizer.language}', task='{tokenizer.task}'")
except Exception as e:
    print(f"Error loading processor for {LANGUAGE} with specific language/task. Trying base: {e}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL_NAME) # Load base
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    print(f"Processor loaded. Explicitly setting language to '{LANGUAGE}' and task to '{TASK}' for tokenizer default state.")
    tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK) # Set defaults if loaded without

# --- 3. Prepare Data for Model (Preprocessing) ---
# The tokenizer here will tokenize the 'sentence' (which can be NL or FR).
# Whisper's tokenizer is multilingual and includes tokens for both.
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("\nPreprocessing synthetic dataset...")
tokenized_datasets = dataset_dict.map(
    prepare_dataset,
    remove_columns=dataset_dict.column_names["train"],
    num_proc=1 # Adjust num_proc based on your system and dataset size
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

# --- 5. Define Evaluation Metric (WER) & Compute Metrics Function ---
wer_metric = evaluate.load("wer")

def compute_metrics(pred: EvalPrediction):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # The tokenizer for decoding should be the one from the processor.
    # Since we are not forcing language globally, the model predicts it.
    # The `tokenizer` variable here refers to the global one defined in step 2.
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if pred_str and label_str:
        print("\n--- DEBUG: Decoded Predictions vs Labels (first sample in batch during Trainer eval) ---")
        print(f"Pred: {pred_str[0]}")
        print(f"Label: {label_str[0]}")

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# --- 6. Load Pre-trained Model ---
print(f"\n--- Loading Pre-trained Model: {BASE_MODEL_NAME} ---")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

# CRITICAL MODIFICATION:
# Do NOT force decoder IDs globally on the model config if you want the model to
# predict the language from mixed input data (e.g., NL audio/NL labels + FR audio/FR labels).
# The model will then decide the language based on the audio.
model.config.forced_decoder_ids = None
model.config.suppress_tokens = [] # Ensure it's an empty list, not None

print(f"Model config forced_decoder_ids explicitly set to None to allow language prediction.")
print(f"model.config.suppress_tokens: {model.config.suppress_tokens}")


# --- 7. Define Training Arguments ---
print(f"\n--- Defining Training Arguments (Target: {NUM_TRAIN_EPOCHS} Epochs, Early Stopping) ---")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    gradient_checkpointing=True,
    fp16=FP16,
    fp16_full_eval=FP16_FULL_EVAL,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=EVAL_BATCH_SIZE_TRAINER,
    predict_with_generate=True, # This tells the trainer to use model.generate for evaluation
    generation_max_length=GENERATION_MAX_LENGTH, # Used by model.generate during trainer eval
    logging_steps=10,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=SAVE_TOTAL_LIMIT,
    logging_dir=f"{OUTPUT_DIR}/logs",
    dataloader_num_workers=0, # Set to 0 for easier debugging; can increase if I/O is a bottleneck
    optim="adamw_torch",
    remove_unused_columns=False, # Important if your dataset has extra columns not used by model
)

# --- 8. Initialize Trainer ---
print("\n--- Initializing Trainer with Early Stopping ---")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"].select(range(min(MAX_SYNTHETIC_EVAL_SAMPLES, len(tokenized_datasets["test"])))),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor, # Pass the full processor (contains tokenizer & feature_extractor)
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
    )]
)

# --- 9. Start Fine-Tuning ---
print(f"\n--- Starting Fine-Tuning (Target: {NUM_TRAIN_EPOCHS} Epochs, Early Stopping Patience: {EARLY_STOPPING_PATIENCE}) ---")
start_time = time.time()

try:
    train_result = trainer.train()
    end_time = time.time()
    training_duration_seconds = end_time - start_time
    print(f"Fine-tuning finished in {training_duration_seconds:.2f} seconds ({(training_duration_seconds/3600):.2f} hours).")

# --- 10. Save Model and Metrics ---
    print(f"\nSaving final model (best model due to load_best_model_at_end=True and EarlyStopping) to {OUTPUT_DIR}")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR) # Save the processor for easy loading later

    metrics = train_result.metrics
    metrics["train_duration_seconds"] = training_duration_seconds
    trainer.log_metrics("train_final", metrics)
    trainer.save_metrics("train_final", metrics)
    print(f"Final training metrics (from train_result): {metrics}")
    if trainer.state.best_metric is not None:
        print(f"Best 'wer' achieved during training: {trainer.state.best_metric:.4f} at step {trainer.state.best_model_checkpoint}")

# --- 11. Manually Evaluate Final Model on the (Synthetic) Test Set ---
    print("\n--- Manually Evaluating the Final Model (from trainer.model) on the Synthetic Test Set ---")
    
    eval_model = trainer.model # This is the best model if load_best_model_at_end=True
    eval_model.eval()
    device = eval_model.device # Get device model is on (e.g., 'cuda:0')

    if len(tokenized_datasets["test"]) > MAX_SYNTHETIC_EVAL_SAMPLES:
        print(f"INFO: Manual evaluation using a subset of {MAX_SYNTHETIC_EVAL_SAMPLES} samples (original size: {len(tokenized_datasets['test'])}).")
        synthetic_eval_dataset_for_manual_eval = tokenized_datasets["test"].shuffle(seed=42).select(range(MAX_SYNTHETIC_EVAL_SAMPLES))
    else:
        synthetic_eval_dataset_for_manual_eval = tokenized_datasets["test"]

    eval_dataloader = DataLoader(
        synthetic_eval_dataset_for_manual_eval,
        batch_size=EVAL_BATCH_SIZE_MANUAL,
        collate_fn=data_collator,
        num_workers=0 # Set to 0 for easier debugging
    )

    all_predictions_ids = []
    all_labels_ids = []
    
    # MODIFICATION for Manual Evaluation:
    # Do NOT force decoder IDs in GenerationConfig if you want model to predict language.
    # The model will use eval_model.config.forced_decoder_ids (which is None) by default.
    # If you LATER want to force a specific language for THIS model during INFERENCE,
    # you can create a GenerationConfig with specific forced_decoder_ids.
    generation_config_manual = GenerationConfig(
        max_length=GENERATION_MAX_LENGTH,
        # forced_decoder_ids=... # REMOVED - let model predict language based on audio
        # suppress_tokens=...   # Can add if needed, but eval_model.config.suppress_tokens will be used by default
    )
    print(f"GenerationConfig for manual eval (no forced language): {generation_config_manual}")
    print(f"Model will use its internal config: forced_decoder_ids={eval_model.config.forced_decoder_ids}, suppress_tokens={eval_model.config.suppress_tokens}")


    print(f"Starting manual evaluation on {device} with batch size {EVAL_BATCH_SIZE_MANUAL}...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0: # Log progress occasionally
                 print(f"  Processing batch {batch_idx + 1}/{len(eval_dataloader)}")
            input_features = batch["input_features"].to(device)
            labels = batch["labels"] # Keep labels on CPU for now, extend list

            generated_ids = eval_model.generate(
                inputs=input_features,
                generation_config=generation_config_manual # This config does not force language
            )
            
            all_predictions_ids.extend(generated_ids.cpu().numpy())
            all_labels_ids.extend(labels.numpy()) # labels are already on CPU from collator
    
    print("Manual evaluation: Decoding predictions and labels...")
    # Ensure correct tokenizer (from processor) is used for decoding
    processed_labels_ids = [
        np.where(label_seq == -100, processor.tokenizer.pad_token_id, label_seq)
        for label_seq in all_labels_ids
    ]

    pred_str = processor.batch_decode(all_predictions_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(processed_labels_ids, skip_special_tokens=True)

    print("\nSample Predictions vs Labels (first 3 from manual eval):")
    for i in range(min(3, len(pred_str))):
        print(f"  Prediction {i+1}: {pred_str[i]}")
        print(f"  Label      {i+1}: {label_str[i]}")
        print("-" * 20)

    print("Manual evaluation: Computing WER...")
    # wer_metric.compute typically returns a float if only one metric is being computed
    wer_score_manual_value = wer_metric.compute(predictions=pred_str, references=label_str)
    if wer_score_manual_value is None: wer_score_manual_value = float('nan')
    
    eval_metrics_to_save_manual = {
        f"eval_wer_manual_final_model": wer_score_manual_value * 100 if not np.isnan(wer_score_manual_value) else float('nan'),
        f"eval_samples_manual_final_model": len(synthetic_eval_dataset_for_manual_eval),
    }
    print(f"Manual evaluation metrics (on final loaded model): {eval_metrics_to_save_manual}")
    
    eval_results_path = Path(OUTPUT_DIR) / "eval_results_manual_final_model.json"
    with open(eval_results_path, "w") as f:
        json.dump(eval_metrics_to_save_manual, f, indent=4)
    
    print(f"Manual evaluation results on synthetic test set (for final loaded model) saved to: {eval_results_path}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\nCUDA cache cleared.")

print(f"\n--- Script complete. Model saved to {OUTPUT_DIR} ---")
print("The model saved should be the one that achieved the best 'wer' during training due to 'load_best_model_at_end=True'.")
print("You can now load your fine-tuned model using:")
print(f"  from transformers import WhisperForConditionalGeneration, WhisperProcessor")
print(f"  model = WhisperForConditionalGeneration.from_pretrained('{OUTPUT_DIR}')")
print(f"  processor = WhisperProcessor.from_pretrained('{OUTPUT_DIR}')")
print("\nIMPORTANT: Now, evaluate this model on your REAL, HELD-OUT JASMIN test sets (NL and FR)")
print("to assess its performance on actual speech.")