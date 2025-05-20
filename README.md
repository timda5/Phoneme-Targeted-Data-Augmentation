Okay, here is the full `README.md` content, incorporating the section to link to a PDF of your thesis stored within the repository.

Remember to:
1.  Replace placeholders like `https://gitlab.com/your-username/your-repository-name` with your actual GitLab URL.
2.  Replace `Damen_Tim_Thesis_ASR_Bias_Mitigation.pdf` with the actual filename of your PDF.
3.  Make sure you've actually added the PDF to a `paper/` subdirectory in your repository and committed/pushed it.

---
```markdown
# Mitigating Bias in Automatic Speech Recognition Through Phoneme-Targeted Data Augmentation

**Author:** Tim Damen
**Affiliation:** Amsterdam University of Applied Sciences
**Contact:** tim.damen@hva.nl
**GitLab Repository:** `https://github.com/timda5/Phoneme-Targeted-Data-Augmentation` 

## Abstract

Automatic Speech Recognition (ASR) systems often exhibit performance disparities across demographic groups, impacting equitable access to voice technologies. This research investigates Phoneme-Targeted Data Augmentation (PTDA) as a technique to mitigate bias between native Dutch-speaking children (NL) and French-speaking children learning Dutch (FR) when using OpenAI’s Whisper Large-v3 and Large-v3-Turbo models. The PTDA approach involved identifying problematic phonemes specific to these groups and generating synthetic speech using Coqui TTS to target these error patterns. The Whisper models were subsequently fine-tuned using this augmented dataset.

A significant constraint of this research iteration was the premature conclusion of the fine-tuning process after only 5 epochs, resulting in models that did not achieve convergence. The results from these under-trained models indicated that PTDA led to a catastrophic degradation in Word Error Rate (WER) and exacerbated bias against FR children for the Whisper Large-v3 model. For the Whisper Large-v3-Turbo model, while mean WER for NL children improved and marginally for FR children, the overall bias (measured by WER difference) also increased, disadvantaging FR children; median WER for the FR group degraded. Phoneme Error Rate (PER) analysis, while showing some targeted improvements, often revealed increased or shifted disparities.

Therefore, under these severely limited training conditions, PTDA was not found to be effective in reducing bias and, in several aspects, worsened performance. The insufficient training duration is identified as the primary factor preventing a conclusive evaluation of PTDA’s potential. This research is ongoing.

## Thesis Document

The complete research methodology, detailed findings, analysis, and discussion for this project are presented in the full thesis document. As the thesis is not yet officially published online, a copy is included in this repository for direct access:

*   **Title:** Mitigating Bias in Automatic Speech Recognition Through Phoneme-Targeted Data Augmentation
*   **Author:** Tim Damen
*   **Affiliation:** Amsterdam University of Applied Sciences
*   **Download/View PDF:** [`./paper/Damen_Tim_Thesis_ASR_Bias_Mitigation.pdf`](./paper/Damen_Tim_Thesis_ASR_Bias_Mitigation.pdf)
  
_Please note: This document reflects the state of the research at the time of its submission. Future updates to the codebase or findings may occur after the thesis completion._

## Research Context & Motivation

ASR systems are vital for applications like voice assistants and real-time captioning. However, they often perform unequally across speaker demographics, creating barriers for groups like non-native speakers or children. This research, conducted in collaboration with RTL as part of the DRAMA project ("Designing Responsible AI Media Applications"), aims to improve ASR fairness, particularly for subtitling applications where equitable performance is crucial for accessibility. The focus is on mitigating bias in Dutch ASR for child speakers.

## Research Questions

**Primary:**
*   How effective is Phoneme-Targeted Data Augmentation in reducing bias between native Dutch-speaking children and French-speaking children learning Dutch in the Whisper ASR system?

**Secondary:**
1.  Which specific Dutch phonemes show the largest recognition performance disparities between native Dutch-speaking children and French-speaking children learning Dutch?
2.  How does the performance of Whisper Large-v3-Turbo compare to Whisper Large-v3 when processing Dutch speech from these two demographic groups?
3.  Can synthetic speech data specifically designed to target problematic phonemes effectively reduce performance disparities between these groups?

## Methodology Overview

The project follows a multi-phase approach:

1.  **Phase 0: Defining Study Populations:** Precisely identifying and extracting speaker cohorts (Native Dutch children and French-speaking children learning Dutch, ages 7-11) from the JASMIN corpus metadata.
2.  **Phase 1: Baseline Performance & Error Analysis:** Establishing baseline ASR performance using Whisper (Large-v3 & Large-v3-Turbo) for both groups and conducting a detailed comparative phoneme-level error analysis to identify specific phonemes contributing to bias.
3.  **Phase 2: Phoneme-Targeted Data Augmentation (PTDA):**
    *   Selecting high-quality voice donors from both original speaker groups.
    *   Sourcing Dutch text sentences rich in the identified problematic phonemes from the OSCAR dataset.
    *   Generating synthetic audio using Coqui TTS (XTTSv2) with cloned voices from both speaker groups, speaking the targeted Dutch sentences. This creates augmented data designed to address phonemic weaknesses.
4.  **Phase 3: Finetuning & Evaluation:** Finetuning the Whisper models with the PTDA dataset and then re-evaluating performance on the original test sets for both groups to assess PTDA's impact on bias and overall ASR accuracy.

## Datasets Used

*   **JASMIN-CGN Corpus:** Source of child speech data.
    *   **Native Dutch-speaking children (NL):** Ages 7-11, L1 (HomeLanguage1) is Dutch, L2 (HomeLanguage2) is Dutch or unspecified.
    *   **French-speaking children learning Dutch (FR):** Ages 7-11, L1 is French, L2 is French (indicating a primarily French linguistic background within the Dutch JASMIN corpus).
    *   **Note:** Access to the JASMIN-CGN corpus must be obtained separately due to its licensing. This repository does not include the corpus data.
*   **OSCAR (Open Super-large Crawled Aggregated coRpus):** Used to source Dutch sentences for PTDA.

## Core Technologies & Frameworks

*   **Python:** 3.10 (as specified in `environment_espnet_py310.yml`)
*   **ASR Model:** OpenAI Whisper (Large-v3 and Large-v3-Turbo)
*   **TTS System:** Coqui TTS (XTTSv2 model)
*   **Key Libraries:** Hugging Face `transformers`, `datasets`, `evaluate`, `pandas`, `numpy`, `librosa`, `torch`, `torchaudio` (see `environment_espnet_py310.yml` for a complete list).
*   **Phonetic Analysis:** `espeak-ng` (via Python wrapper or custom scripts), Kaldi (for `align-text`).
*   **Corpus Processing:** Scripts adapted from [karkirowle/relative_phoneme_analysis](https://github.com/karkirowle/relative_phoneme_analysis) for G2P and PER analysis.

## Setup Instructions

### Prerequisites
*   Conda (recommended for environment management)
*   Git
*   `ffmpeg` (required by `librosa` and `datasets` for audio processing):
    ```bash
    # On Ubuntu/Debian
    sudo apt update && sudo apt install ffmpeg
    # On macOS (using Homebrew)
    brew install ffmpeg
    # On Windows (using Chocolatey or manual install)
    choco install ffmpeg # or download from ffmpeg.org and add to PATH
    ```
*   For Kaldi `align-text`: A working Kaldi installation is required. Ensure `align-text` is in your PATH.

### 1. Clone the Repository
```bash
git clone https://github.com/timda5/Phoneme-Targeted-Data-Augmentation
cd Phoneme-Targeted-Data-Augmentation
```

### 2. Create and Activate Conda Environment
This project uses a Conda environment defined in `environment_espnet_py310.yml`.
```bash
conda env create -f environment_espnet_py310.yml
conda activate espnet_py310
```
Alternatively, if you prefer to set up manually or use a `requirements.txt` (if provided):
```bash
# Example for manual setup (adjust Python version if needed)
conda create -n ptda_env python=3.10
conda activate ptda_env
# pip install -r requirements.txt # If a requirements.txt is available
# Or install key packages manually:
# pip install torch torchvision torchaudio
# pip install transformers datasets evaluate pandas numpy librosa coqui-tts espeakng kaldiio
```

### 3. Obtain and Prepare Datasets
*   **JASMIN-CGN Corpus:**
    *   This corpus is **not** included in the repository and must be obtained separately from its official distributor (e.g., CLST Radboud University Nijmegen or ELRA).
    *   Once obtained, ensure the directory structure is accessible by the scripts. You will likely need to configure paths within the scripts (e.g., in a `config.py` file or at the top of relevant scripts) to point to your local JASMIN-CGN installation.
    *   The scripts in "Phase 0" will process its metadata.
*   **OSCAR Dataset:**
    *   The Dutch portion of the OSCAR dataset is used. Scripts in "Phase 2" (specifically `oscar.py`) will process this data.
    *   The `oscar.py` script is designed to stream or download relevant parts as needed, typically using the Hugging Face `datasets` library. Ensure you have an internet connection when running this script for the first time for a new dataset split.

## Pipeline Overview & Scripts

The project is structured into several phases, implemented by the scripts listed below. It's generally recommended to run them in the order presented.

### Phase 0: Defining Study Populations
1.  **`identify_speaker_groups.py`**
    *   **Purpose:** Identifies and extracts speaker codes for the two target child speaker groups (NL and FR, ages 7-11) from the JASMIN corpus metadata.
    *   **Inputs:** Path to JASMIN metadata files (e.g., `nl/speakers.txt`, `vl/speakers.txt`). _Configuration of this path is likely needed within the script._
    *   **Outputs:** Text files containing RegionSpeaker codes (speaker IDs) for each group (e.g., `pure_dutch_children_7_11_codes.txt`, `pure_french_children_7_11_codes.txt`). These files are crucial for subsequent scripts and are typically saved in a `data/processed_metadata/` or similar directory.
    *   **Key Actions:** Loads metadata, cleans column names, filters by age and home languages (L1/L2).

### Phase 1: Baseline Performance and Identifying Problematic Phonemes
2.  **`transcribe_jasmin_baseline.py`** (or similar name for Whisper transcription)
    *   **Purpose:** Transcribes speech from both defined child speaker groups using a specified Whisper model (e.g., Large-v3 or Large-v3-Turbo) to establish baseline performance.
    *   **Inputs:** Speaker code lists generated by `identify_speaker_groups.py`; Path to JASMIN audio (`.wav`) and reference transcription (`.ort`) files; Whisper model name (e.g., `openai/whisper-large-v3`). _Paths to JASMIN data and model names may need configuration._
    *   **Outputs:** Group-specific directories (e.g., `output/transcriptions/baseline/PureDutchChildren_7_11_L3/`) containing detailed JSON files (with ASR hypotheses, word-level timestamps) and simple text transcriptions (`.txt`).
    *   **Key Actions:** Iterates through speakers, locates audio/ORT files, robustly parses ORT files, runs Whisper transcription. *This script is run separately for each group and for each Whisper model variant.*

3.  **`prepare_kaldi_wer_files.ipynb`** (or `WER_Kaldi.ipynb`)
    *   **Purpose:** Processes the JSON outputs from the Whisper transcription script to prepare files for WER calculation and alignment using Kaldi tools.
    *   **Inputs:** Group-specific JSON output directories from the previous script. _Paths to these directories will need to be set in the notebook._
    *   **Outputs:** Pairs of Kaldi-formatted reference and hypothesis text files for each group (e.g., `kaldi_io/ref_PureDutchChildren.txt`, `kaldi_io/hyp_PureDutchChildren.txt`).
    *   **Key Actions:** Applies hypothesis filtering, text normalization (uppercase, punctuation removal, JASMIN annotation stripping).

4.  **Kaldi `align-text` (Terminal Command)**
    *   **Purpose:** Performs word-by-word alignment between reference and hypothesis transcriptions.
    *   **Inputs:** Kaldi-formatted `ref_<group>.txt` and `hyp_<group>.txt` files from the previous step.
    *   **Command Example (run from the directory containing the ref/hyp files, or adjust paths):**
        ```bash
        align-text ark:ref_PureDutchChildren.txt ark:hyp_PureDutchChildren.txt ark,t:- > aligned_PureDutchChildren.txt
        align-text ark:ref_PureFrenchChildren.txt ark:hyp_PureFrenchChildren.txt ark,t:- > aligned_PureFrenchChildren.txt
        ```
    *   **Outputs:** Detailed alignment files (e.g., `aligned_PureDutchChildren.txt`).
    *   **Key Actions:** Run separately for each speaker group and model variant.

5.  **`generate_per_utt.py`**
    *   **Purpose:** Converts Kaldi's `align-text` output into the 'per_utt' format required for detailed phoneme error analysis (compatible with `karkirowle/relative_phoneme_analysis` tools).
    *   **Inputs:** Aligned output files from `align-text` (e.g., `aligned_PureDutchChildren.txt`). _Input file paths need configuration._
    *   **Outputs:** Structured per-utterance error files (e.g., `per_utt_output/per_utt_final_PureDutchChildren.txt`) containing CSID (Correct, Substitution, Insertion, Deletion) information.
    *   **Key Actions:** Processes alignment data for each group independently.

6.  **`PER_Analysis.py`**
    *   **Purpose:** Conducts the core comparative phoneme-level error analysis between the NL and FR child groups to identify Dutch phonemes with the largest recognition performance disparities.
    *   **Inputs:** `per_utt_final_<group>.txt` files for both groups. _Input file paths need configuration._
    *   **Outputs:** Reports/CSVs detailing PERs, error differences, frequencies, and "Importance Scores" for phonemes (e.g., `analysis_results/phoneme_disparities.csv`). Identifies target phonemes for PTDA.
    *   **Key Actions:** Uses a custom `MyWERDetails` class (potentially with `espeak-ng` for G2P of Out-Of-Vocabulary words) for processing. Directly addresses secondary RQ1.

### Phase 2: Phoneme-Targeted Data Augmentation (PTDA)
7.  **`analyze_utt_errors.py`**
    *   **Purpose:** Identifies the best-performing original speakers from both NL and FR child groups (based on low per-utterance WERs from baseline analysis) to serve as voice donors for TTS.
    *   **Inputs:** `per_utt_final_<group>.txt` files for both groups (from baseline analysis). _Input file paths need configuration._
    *   **Outputs:** Text files listing 'best voice donor' utterance IDs and corresponding audio file paths for each group (e.g., `data/voice_donors/lowest_error_utt_ids_nl.txt`, `data/voice_donors/lowest_error_utt_ids_fr.txt`).
    *   **Key Actions:** Selects top N lowest WER utterance IDs from each group.

8.  **`oscar.py`**
    *   **Purpose:** Scrapes the OSCAR dataset for Dutch text sentences rich in the phonemes identified by `PER_Analysis.py` as problematic.
    *   **Inputs:** List of target phonemes (e.g., from `analysis_results/phoneme_disparities.csv`); Access to the OSCAR Dutch dataset (via Hugging Face `datasets`). _May require configuration of target phonemes and output paths._
    *   **Outputs:** Text files containing phoneme-rich Dutch sentences (e.g., `data/augmentation_text/output_nl_rich_sentences.txt`, `data/augmentation_text/output_fr_rich_sentences.txt`).
    *   **Key Actions:** Streams OSCAR sentences, cleans them, uses `espeak-ng` for G2P of Dutch text, selects sentences with high density of target phonemes.

9.  **`clean_output_files.py`**
    *   **Purpose:** Performs additional cleaning and deduplication on the phoneme-rich Dutch sentences harvested by `oscar.py`.
    *   **Inputs:** Output files from `oscar.py`. _Input/output paths need configuration._
    *   **Outputs:** Cleaned and deduplicated sentence files (e.g., `data/augmentation_text/cleaned_nl_rich_sentences.txt`).
    *   **Key Actions:** Ensures high-quality, diverse text input for TTS.

10. **`synthesize_coqui_audio.py`** (or similar for Coqui TTS)
    *   **Purpose:** Generates synthetic audio for PTDA using Coqui TTS (XTTSv2 model).
    *   **Inputs:** Cleaned phoneme-rich Dutch sentences; Paths to audio files of 'best voice donor' speakers (from `analyze_utt_errors.py` and JASMIN corpus) for voice cloning. _Paths to sentences, donor audio, and output directories need configuration._
    *   **Outputs:** Synthetic audio files (`.wav`) saved in group-specific directories (e.g., `data/synthetic_audio_coqui/nl_voices/`, `data/synthetic_audio_coqui/fr_voices/`); Metadata file (e.g., `transcript.csv`) linking synthetic audio to source speaker ID and the Dutch sentence text.
    *   **Key Actions:** Clones voices of selected NL and FR child speakers; Synthesizes targeted Dutch sentences using these cloned voices (language for synthesis is 'nl'). This generates the PTDA dataset.

### Phase 3: Finetuning Whisper and Evaluating PTDA Impact
11. **Whisper Finetuning (e.g., `finetune_whisper.py` or `finetune_whisper.ipynb`)**
    *   **Purpose:** Finetunes the baseline Whisper models (Large-v3 and Large-v3-Turbo) using the PTDA dataset. This typically involves a script utilizing the Hugging Face `Trainer` API.
    *   **Inputs:** The synthetic audio dataset generated by `synthesize_coqui_audio.py` (audio files and `transcript.csv`); (Optionally) Original training data if performing mixed training; Pre-trained Whisper model checkpoints (e.g., `openai/whisper-large-v3`); Training hyperparameters (batch size, learning rate, number of epochs, etc. – see thesis for details).
    *   **Outputs:** Finetuned Whisper model checkpoints (saved to a specified output directory, e.g., `models/whisper_large_v3_finetuned_ptda/`).
    *   **Key Actions:** Prepares dataset for `transformers` (loading audio, tokenizing text); Defines training arguments and trainer; Runs training loop. **Note: In the current research iteration, this was limited to 5 epochs.**

12. **Re-evaluation Pipeline (Repeat Scripts 2-6)**
    *   **Purpose:** To evaluate the performance of the newly finetuned Whisper models on the original JASMIN test sets for both NL and FR child groups.
    *   **Process:**
        1.  Run `transcribe_jasmin_baseline.py` (or a similar script, possibly renamed e.g., `transcribe_jasmin_finetuned.py`) using the **finetuned Whisper model paths** to get new transcriptions. Ensure outputs are saved to a new directory (e.g., `output/transcriptions/finetuned/`).
        2.  Run `prepare_kaldi_wer_files.ipynb` on these new transcriptions.
        3.  Run Kaldi `align-text` on the new ref/hyp files.
        4.  Run `generate_per_utt.py` on the new aligned files.
        5.  Run `PER_Analysis.py` on the new per_utt files.
    *   **Contribution:** This allows for comparison of WER and PER metrics between the baseline and finetuned models for both groups, assessing the impact of PTDA. Results should be compared against the baseline results.

## Running the Pipeline

1.  **Ensure Setup is Complete:**
    *   Conda environment (`espnet_py310`) is activated.
    *   Datasets (JASMIN-CGN) are obtained and accessible. Paths within scripts may need to be configured.
    *   Kaldi `align-text` is installed and in PATH.
    *   `ffmpeg` is installed.

2.  **Execute Scripts Sequentially:** Follow the phases and script numbers outlined above.
    *   Start with `identify_speaker_groups.py`.
    *   Proceed through Phase 1 scripts for baseline analysis (for each Whisper model variant: Large-v3 and Large-v3-Turbo).
    *   Execute Phase 2 scripts to generate the PTDA dataset.
    *   Perform Whisper finetuning (Script 11).
    *   Re-run the evaluation pipeline (Script 12, which reuses Scripts 2-6) using the finetuned models.

3.  **Configuration:**
    *   Many scripts will require configuration of paths to data, models, and output directories. Check the beginning of each Python script or Jupyter Notebook for configurable variables.
    *   Consider creating a central `config.py` file to manage paths if preferred.

## Summary of Current Findings (Based on 5-Epoch Finetuning)

**Crucial Context:** The finetuning process for this iteration of the research was concluded after only **5 epochs** due to computational/time constraints, despite being configured for 20. This resulted in models that **did not achieve convergence** (e.g., training loss of 0.4919 for Large-v3), significantly impacting the results.

*   **Whisper Large-v3 (Finetuned with PTDA):**
    *   Led to a catastrophic degradation in Word Error Rate (WER) for both NL and FR children.
    *   Exacerbated bias against FR children (Mean WER difference FR-NL changed from -0.76% to +17.47%).
    *   Phoneme Error Rates (PERs) also showed increased bias and degradation.
*   **Whisper Large-v3-Turbo (Finetuned with PTDA):**
    *   Mean WER for NL children improved considerably.
    *   Mean WER for FR children improved marginally, but their median WER degraded.
    *   The overall bias (Mean WER difference FR-NL) increased, disadvantaging FR children (changed from -1.07% to +5.51%).
    *   PER analysis showed mixed results: some targeted phonemes improved in absolute terms for both groups (notably /y/ showed reduced bias), but for many others, the relative performance gap widened or shifted to further disadvantage the FR group.

**Conclusion from this Iteration:** Under these severely limited training conditions (5 epochs), PTDA was not found to be effective in reducing bias and, in several aspects, worsened performance or amplified disparities. The insufficient training duration is identified as the primary factor preventing a conclusive evaluation of PTDA’s potential.

## Limitations

*   **Severely Limited Training Duration:** The most significant limitation. Models were under-trained, preventing a fair evaluation of PTDA's potential.
*   **Synthetic Speech Quality:** While Coqui XTTSv2 is advanced, the naturalness and precise phonetic realization of synthetic child speech (especially accented child speech mimicking L2 learners) remain challenging and may not perfectly match real speech.
*   **Dataset Specificity:** Results are based on the JASMIN-CGN corpus and the specific characteristics of its NL and FR child speaker groups. Generalizability to other datasets or demographic groups requires further investigation.
*   **Scope of Phoneme Analysis:** Primarily PER-based, focusing on phoneme substitutions, insertions, and deletions. Other acoustic-phonetic features (e.g., prosody, intonation) that might contribute to bias were not explicitly targeted by the PTDA.
*   **Voice Donor Selection:** While based on low WER, the "best" voice donors might not fully represent the diversity of speech patterns within each group, potentially limiting the effectiveness of voice cloning for all target phonetic contexts.

## Future Work

*   **Ensure Sufficient Training and Convergence:** **(Highest Priority)** Re-run finetuning for the full configured epochs (e.g., 20) or until convergence is observed on a validation set. This is essential for a valid evaluation of PTDA.
*   **Refine Synthetic Data Generation:**
    *   Explore more advanced TTS systems or techniques for higher fidelity child speech and more nuanced control over phoneme realizations, potentially including models that can better capture L2 accents.
    *   Experiment with the amount and diversity of synthetic data.
*   **Iterative Refinement & Hyperparameter Exploration:** Once full training is possible, systematically explore a wider range of finetuning hyperparameters (learning rate, batch size, schedulers) and model adaptation techniques (e.g., LoRA, prompt tuning).
*   **Investigate Model-Specific Responses:** Further analyze why Whisper Large-v3 and Large-v3-Turbo respond differently to PTDA. This might involve examining internal model representations or attention patterns.
*   **Expanded Error Analysis:** Incorporate more detailed acoustic-phonetic analysis beyond PER to understand the nature of remaining errors and biases.
*   **Validation Set:** Implement a proper validation set during finetuning for early stopping and hyperparameter tuning to prevent overfitting and guide model selection.
*   **Mixed Data Training:** Experiment with mixing the PTDA data with a portion of the original child speech data during finetuning, rather than using only synthetic data, to potentially improve robustness.

## How to Cite

If you use this work, code, or findings in your research, please consider citing the associated thesis document (included in this repository) and/or this GitLab repository.

**Thesis Document:**
Damen, T. (2025). *Mitigating Bias in Automatic Speech Recognition Through Phoneme-Targeted Data Augmentation*. [Master's Thesis/Research Project Report]. Amsterdam University of Applied Sciences. (Available in the `paper/` directory of this repository).

**GitLab Repository:**
```
Damen, T. (2025). Mitigating Bias in Automatic Speech Recognition Through Phoneme-Targeted Data Augmentation. [Software]. GitLab. https://github.com/timda5/Phoneme-Targeted-Data-Augmentation
```

## Acknowledgements

*   This research was conducted as part of the DRAMA project ("Designing Responsible AI Media Applications") in collaboration with RTL Netherlands.
*   Acknowledgement to the developers of the JASMIN-CGN corpus for providing valuable child speech data.
*   Thanks to the open-source community for tools like OpenAI Whisper, Coqui TTS, Hugging Face Transformers, Kaldi, and espeak-ng.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
```
