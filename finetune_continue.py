# --- Environment Variable Setup (Set these BEFORE importing PyTorch) ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Of je gewenste GPU index
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
    TrainingArguments, # Expliciet importeren voor duidelijkheid
    GenerationConfig
)
import evaluate # Hugging Face Evaluate library
import sys # Voor afsluiten als er geen data is
import time # Importeer time module voor timing
import numpy as np # Voor handmatige evaluatie
from torch.utils.data import DataLoader # Voor handmatige evaluatie
import json # Voor opslaan eval_results.json
import shutil # Niet direct gebruikt, maar kan handig zijn voor logbeheer

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
    # Overweeg af te sluiten als GPU essentieel is
    # sys.exit("CUDA not available. Exiting.")


# --- 0. Configuration ---
PRETRAINED_MODEL_PATH_FOR_CONTINUATION = "whisper-finetuned-synthetic-only-large-v3" # Pad naar je model na 5 epochs
LANGUAGE = "nl" # Primaire taal voor tokenizer setup
TASK = "transcribe"

# NIEUWE output directory voor de voortgezette training
OUTPUT_DIR_CONTINUED = "whisper-finetuned-synthetic-continued-large-v3"
Path(OUTPUT_DIR_CONTINUED).mkdir(parents=True, exist_ok=True) # Maak de directory als deze nog niet bestaat

SYNTHETIC_AUDIO_BASE_DIR = Path('synthetic_audio_coqui')
NL_SYNTHETIC_METADATA_CSV = SYNTHETIC_AUDIO_BASE_DIR / 'nl' / 'nl_metadata.csv'
FR_SYNTHETIC_METADATA_CSV = SYNTHETIC_AUDIO_BASE_DIR / 'fr' / 'fr_metadata.csv'

# Training Hyperparameters
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4 # Gebruikt voor handmatige evaluatie DataLoader
NUM_ADDITIONAL_EPOCHS = 15 # AANTAL EXTRA EPOCHS om te trainen
LEARNING_RATE = 5e-6       # Behoud dezelfde learning rate, of pas aan indien gewenst
WARMUP_STEPS = 50          # Optimizer wordt gereset, dus warmup is weer van toepassing
GRADIENT_ACCUMULATION_STEPS = 4
FP16 = torch.cuda.is_available()
GENERATION_MAX_LENGTH = 225 # Max lengte voor gegenereerde sequenties
MAX_SYNTHETIC_EVAL_SAMPLES = 1000 # Max samples voor synthetische validatie om geheugen te besparen

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

if len(raw_dataset) < 10: # Arbitraire grens voor een zinvolle split
    print("Synthetic dataset too small to split effectively. Using all for training and testing (not ideal).")
    # Overweeg om de test set leeg te laten of een zeer kleine subset te gebruiken indien nodig.
    # Voor nu gebruiken we alles voor beide als het erg klein is.
    dataset_dict = DatasetDict({'train': raw_dataset, 'test': raw_dataset.select(range(min(len(raw_dataset), 5))) if len(raw_dataset)>0 else raw_dataset})
else:
    train_test_split = raw_dataset.train_test_split(test_size=0.1, seed=42) # Gebruik een seed voor reproduceerbaarheid
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=16000))
print("\nSynthetic dataset prepared and split:")
print(dataset_dict)

# --- 2. Load Processor (Feature Extractor and Tokenizer) ---
print(f"\n--- Loading Processor from {PRETRAINED_MODEL_PATH_FOR_CONTINUATION} ---")
try:
    # Probeer de processor te laden van het pad waar het gefinetunede model is opgeslagen.
    # Dit zorgt ervoor dat eventuele aanpassingen aan de tokenizer/processor (zoals toegevoegde tokens) meegenomen worden.
    feature_extractor = WhisperFeatureExtractor.from_pretrained(PRETRAINED_MODEL_PATH_FOR_CONTINUATION)
    tokenizer = WhisperTokenizer.from_pretrained(PRETRAINED_MODEL_PATH_FOR_CONTINUATION)
    processor = WhisperProcessor.from_pretrained(PRETRAINED_MODEL_PATH_FOR_CONTINUATION)
    
    # Het kan nodig zijn om taal en taak expliciet opnieuw in te stellen als de `processor.push_to_hub`
    # of `save_pretrained` dit niet perfect heeft opgeslagen voor Whisper.
    # Voor Whisper modellen wordt dit vaak intern beheerd door `forced_decoder_ids` in de `generate` stap,
    # maar het is goed om de tokenizer's prefix tokens correct te hebben voor consistentie.
    print(f"Processor loaded. Explicitly setting language to '{LANGUAGE}' and task to '{TASK}' for tokenizer.")
    tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK) # Herbevestig voor de zekerheid

except OSError: # OSError kan optreden als niet alle processor bestanden aanwezig zijn
    print(f"Warning: Could not load full processor from {PRETRAINED_MODEL_PATH_FOR_CONTINUATION}. "
          f"Attempting to load feature extractor and tokenizer separately, then combine.")
    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(PRETRAINED_MODEL_PATH_FOR_CONTINUATION)
        tokenizer = WhisperTokenizer.from_pretrained(PRETRAINED_MODEL_PATH_FOR_CONTINUATION, language=LANGUAGE, task=TASK)
        processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    except Exception as e_fallback:
        print(f"Error loading processor components from fine-tuned path or setting language/task. "
              f"Critical error, cannot proceed. Check processor files in '{PRETRAINED_MODEL_PATH_FOR_CONTINUATION}'. Error: {e_fallback}")
        sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred loading the processor: {e}")
    sys.exit(1)


# --- 3. Prepare Data for Model (Preprocessing) ---
def prepare_dataset(batch):
    # Laad audio
    audio = batch["audio"]
    # Bereken input features van de audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # Tokenize de target text
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("\nPreprocessing synthetic dataset...")
tokenized_datasets = dataset_dict.map(
    prepare_dataset,
    remove_columns=dataset_dict.column_names["train"], # Verwijder originele kolommen
    num_proc=1 # Gebruik 1 proces voor stabiliteit; verhoog als veilig en dataset groot is
)
print("Synthetic dataset preprocessed.")
print(tokenized_datasets)


# --- 4. Define Data Collator ---
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Vervang padding token id in de labels met -100 zodat ze niet meegenomen worden in de loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Als alle sequenties beginnen met BOS token, strip het. (Whisper specifiek)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- 5. Define Evaluation Metric (WER) ---
wer_metric = evaluate.load("wer")

# --- 6. Load Fine-tuned Model (van de vorige 5 epochs) ---
print(f"\n--- Loading Fine-tuned Model from: {PRETRAINED_MODEL_PATH_FOR_CONTINUATION} ---")
try:
    model = WhisperForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_PATH_FOR_CONTINUATION)
except Exception as e:
    print(f"Could not load model from {PRETRAINED_MODEL_PATH_FOR_CONTINUATION}. Error: {e}")
    sys.exit(1)

# Ensure model.config.suppress_tokens is a list for the print statement and any other direct use
if model.config.suppress_tokens is None:
    print("INFO: model.config.suppress_tokens was None. Setting to an empty list [].")
    model.config.suppress_tokens = []

# The model.config.forced_decoder_ids might be None if it was null in config.json or generation_config.json
# This is fine as long as it's correctly handled during generation.
print(f"Model config after loading: forced_decoder_ids={model.config.forced_decoder_ids}, suppress_tokens length={len(model.config.suppress_tokens)}")



# --- 7. Define Training Arguments for Continued Training ---
print("\n--- Defining Training Arguments for Continued Training ---")
training_args_continued = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR_CONTINUED,  # NIEUWE output directory
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE, # De optimizer wordt gereset, dus LR en warmup zijn weer van toepassing
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_ADDITIONAL_EPOCHS, # AANTAL EXTRA EPOCHS
    fp16=FP16,
    gradient_checkpointing=True, # Kan helpen met geheugengebruik, maar maakt training langzamer
    logging_dir=f"{OUTPUT_DIR_CONTINUED}/logs", # NIEUWE logging directory
    logging_strategy="steps",    # Log elke logging_steps
    logging_steps=25,
    eval_strategy="no",          # Geen evaluatie tijdens deze voortgezette training
                                 # Zet op "epoch" als je tussentijds wilt evalueren op de test set
    save_strategy="epoch",       # NU WEL CHECKPOINTS OPSLAAN (elke epoch)
    save_total_limit=3,          # Bewaar de laatste 3 checkpoints (optioneel, voor ruimtebesparing)
    # load_best_model_at_end=False, # Alleen relevant als eval_strategy niet "no" is
    # report_to=["tensorboard"],   # Uncomment als je TensorBoard wilt gebruiken
    dataloader_num_workers=0,    # Zet op >0 als je data loading bottlenecks hebt en het OS het toelaat
    optim="adamw_torch",         # Aanbevolen optimizer
    remove_unused_columns=False, # Belangrijk voor datasets gemapt met .map()
)

# --- 8. Initialize Trainer for Continued Training ---
trainer_continued = Seq2SeqTrainer(
    args=training_args_continued,
    model=model, # Het geladen model (na 5 epochs)
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"], # De test set voor eventuele evaluatie
    data_collator=data_collator,
    tokenizer=processor, # Geef de volledige processor mee, niet alleen feature_extractor
    # compute_metrics=compute_metrics_function, # Als je eval_strategy != "no" hebt, definieer dan een compute_metrics functie
)

# --- 9. Start Continued Fine-Tuning ---
print(f"\n--- Starting Continued Fine-Tuning for {NUM_ADDITIONAL_EPOCHS} additional epochs ---")
start_time_continued = time.time()

try:
    # We starten een "nieuwe" training sessie met het reeds gedeeltelijk getrainde model.
    # De optimizer en scheduler beginnen opnieuw.
    train_result_continued = trainer_continued.train()
    end_time_continued = time.time()
    training_duration_seconds_continued = end_time_continued - start_time_continued
    print(f"Continued fine-tuning finished in {training_duration_seconds_continued:.2f} seconds ({(training_duration_seconds_continued/3600):.2f} hours).")

    # --- 10. Save Final Model and Metrics from Continued Training ---
    # Het model wordt ook opgeslagen door de save_strategy ("epoch"), dit is een expliciete save van de allerlaatste state.
    final_model_save_path = Path(OUTPUT_DIR_CONTINUED) / "final_model" # Aparte submap voor duidelijkheid
    print(f"\nSaving final model (after {5 + NUM_ADDITIONAL_EPOCHS} total effective epochs) to {final_model_save_path}")
    trainer_continued.save_model(output_dir=str(final_model_save_path))
    # Sla ook de processor op met het model
    processor.save_pretrained(str(final_model_save_path))


    metrics_continued = train_result_continued.metrics
    metrics_continued["train_duration_seconds"] = training_duration_seconds_continued
    # Log de metrics met een onderscheidende naam
    trainer_continued.log_metrics("train_continued_run", metrics_continued)
    trainer_continued.save_metrics("train_continued_run", metrics_continued) # Slaat op in OUTPUT_DIR_CONTINUED
    print(f"Continued training metrics: {metrics_continued}")

    # --- 11. Manually Evaluate Final Model on the (Synthetic) Test Set ---
    print("\n--- Manually Evaluating the Final Model on the Synthetic Test Set ---")
    
    # Gebruik het model dat net is getraind en mogelijk opgeslagen in een checkpoint of `final_model_save_path`
    # Voor deze evaluatie gebruiken we `trainer_continued.model` wat de state in het geheugen is.
    # Als je wilt evalueren vanaf een specifiek checkpoint, laad dat dan eerst.
    
    eval_model = trainer_continued.model # of laad vanaf final_model_save_path als je dat wilt
    eval_model.eval() # Zet model in evaluatiemodus
    device = eval_model.device # Gebruik het device waar het model op staat

    if len(tokenized_datasets["test"]) > MAX_SYNTHETIC_EVAL_SAMPLES:
        print(f"INFO: Using a subset of {MAX_SYNTHETIC_EVAL_SAMPLES} samples for synthetic evaluation (original size: {len(tokenized_datasets['test'])}).")
        synthetic_eval_dataset_for_manual_eval = tokenized_datasets["test"].shuffle(seed=42).select(range(MAX_SYNTHETIC_EVAL_SAMPLES))
    else:
        synthetic_eval_dataset_for_manual_eval = tokenized_datasets["test"]

    eval_dataloader = DataLoader(
        synthetic_eval_dataset_for_manual_eval,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=data_collator,
        num_workers=0 # Zet op 0 voor eenvoud, kan verhoogd worden
    )

    all_predictions_ids = []
    all_labels_ids = []

    generation_config = GenerationConfig.from_pretrained(PRETRAINED_MODEL_PATH_FOR_CONTINUATION) # Start met config van basis
    # Pas aan indien nodig voor deze specifieke generatie
    generation_config.language = LANGUAGE # Zorg dat taal correct is ingesteld voor Whisper
    generation_config.task = TASK
    generation_config.max_length = GENERATION_MAX_LENGTH

    # --- CRITICAL: Ensure forced_decoder_ids are set for Whisper generation ---
    # `processor.get_decoder_prompt_ids` returns a list of tuples, e.g. [(50259, 50359)]
    # `GenerationConfig` expects a list of lists for forced_decoder_ids
    # If generation_config.json had "forced_decoder_ids": null, this will be None.
    # Or if it loaded from a base model without them set for this specific task.
    
    # Check and set forced_decoder_ids
    # The processor's tokenizer should be set to the correct language and task
    # tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK) # Was done earlier
    
    # Default to no_timestamps=True is common for ASR if not explicitly needing timestamps
    decoder_prompt_ids_tuples = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK, no_timestamps=True)
    generation_config.forced_decoder_ids = [list(ids) for ids in decoder_prompt_ids_tuples]
    
    # Ensure suppress_tokens is a list in generation_config as well
    if generation_config.suppress_tokens is None:
        generation_config.suppress_tokens = []
    # You can add specific tokens to suppress if needed, e.g., generation_config.suppress_tokens.extend([processor.tokenizer.no_timestamps_token_id])

    print(f"Starting manual evaluation on {device} with batch size {EVAL_BATCH_SIZE}...")
    print(f"Generation config for manual eval: language='{generation_config.language}', task='{generation_config.task}', max_length={generation_config.max_length}, forced_decoder_ids={generation_config.forced_decoder_ids}, suppress_tokens={generation_config.suppress_tokens}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            print(f"  Processing batch {batch_idx + 1}/{len(eval_dataloader)}")
            input_features = batch["input_features"].to(device)
            labels = batch["labels"] # Blijft op CPU voor nu

            # Genereer output IDs
            generated_ids = eval_model.generate(
                inputs=input_features,
                generation_config=generation_config
            )
            
            all_predictions_ids.extend(generated_ids.cpu().numpy())
            all_labels_ids.extend(labels.numpy()) # labels waren al numpy-compatibel
    
    print("Manual evaluation: Decoding predictions and labels...")
    # Process labels: vervang -100 met pad_token_id voor decoding
    processed_labels_ids = [
        np.where(label_seq == -100, processor.tokenizer.pad_token_id, label_seq)
        for label_seq in all_labels_ids
    ]

    pred_str = processor.batch_decode(all_predictions_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(processed_labels_ids, skip_special_tokens=True)

    # Debug: print enkele voorbeelden
    print("\nSample Predictions vs Labels (first 3):")
    for i in range(min(3, len(pred_str))):
        print(f"  Prediction {i+1}: {pred_str[i]}")
        print(f"  Label      {i+1}: {label_str[i]}")
        print("-" * 20)


    print("Manual evaluation: Computing WER...")
    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)
    if wer_score is None: wer_score = float('nan') # Handle geval waar metric None teruggeeft
    
    eval_metrics_to_save = {
        f"eval_wer_synthetic_test": wer_score * 100 if not np.isnan(wer_score) else float('nan'), # Maak percentage
        f"eval_samples_synthetic_test": len(synthetic_eval_dataset_for_manual_eval),
    }
    # Log metrics naar de console en eventuele loggers (Tensorboard, W&B)
    trainer_continued.log_metrics("eval_synthetic_continued", eval_metrics_to_save)
    
    # Sla de evaluatieresultaten op in de NIEUWE output directory
    eval_results_path = Path(OUTPUT_DIR_CONTINUED) / "eval_results_synthetic_test.json"
    with open(eval_results_path, "w") as f:
        json.dump(eval_metrics_to_save, f, indent=4)
    
    print(f"Manual evaluation results on synthetic test set (saved to {eval_results_path}): {eval_metrics_to_save}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\nCUDA cache cleared.")

print(f"\n--- Script complete. Continued model training finished. ---")
print(f"Checkpoints and logs for continued training are in: {OUTPUT_DIR_CONTINUED}")
print(f"The final model state from this run is saved in: {Path(OUTPUT_DIR_CONTINUED) / 'final_model'}") # Pad naar expliciet opgeslagen model
print("Vergeet niet dit model ook te evalueren op je ECHTE JASMIN test sets (NL en FR).")