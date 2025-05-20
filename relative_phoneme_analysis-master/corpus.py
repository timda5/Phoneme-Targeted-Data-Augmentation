import itertools
# Removed 'text' import as it seems unused if alternative_cmudict is primary
# import text
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
# Ensure per_utils is importable (e.g., in the same directory or PYTHONPATH)
try:
    from per_utils import per_phoneme_per, align_sequences, afer
except ImportError:
    print("Error: Could not import from per_utils.py. Ensure it's accessible.")
    # Handle the error appropriately, maybe exit or raise
    import sys
    sys.exit(1)
import os
import html
from typing import List, Tuple
# Ensure utils is importable
try:
    from utils import HParam
except ImportError:
    print("Error: Could not import HParam from utils.py. Ensure it's accessible.")
    import sys
    sys.exit(1)
import re
import traceback # Import traceback for detailed error printing


class AlternativeCMUDict():

    def __init__(self, location: str, conf):
        """
        Basically an alternative CMUDict parser because the original one is most likely an overkill
        """
        print(f"DEBUG: Initializing AlternativeCMUDict with path: {location}") # DEBUG
        self.conf = conf
        self.cmudict = {} # Initialize as empty dict
        self.n = 0 # Lookup counter
        self.oov = 0 # OOV counter

        try:
            with open(location, encoding='utf-8') as file:
                temp_dict = {}
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line or line.startswith('#'): # Skip empty lines and comments
                        continue
                    # Use regex to split on the first sequence of whitespace
                    parts = re.split(r'\s+', line, maxsplit=1)
                    if len(parts) == 2:
                        word = parts[0].upper() # Normalize word to uppercase
                        pronunciation = parts[1].strip()
                        # Basic validation: Check if pronunciation contains letters (more robust than checking for non-digits)
                        if any(c.isalpha() for c in pronunciation):
                            temp_dict[word] = pronunciation
                        else:
                            print(f"Warning: Skipping potentially invalid pronunciation in dictionary line {line_num}: '{line}'")
                    elif len(parts) == 1:
                         word = parts[0].upper()
                         temp_dict[word] = "" # Assign empty string for words without pronunciation
                         # print(f"Warning: Word '{word}' found with no pronunciation in dictionary line {line_num}.")
                    else:
                        print(f"Warning: Skipping malformed dictionary line {line_num}: '{line}'")
                self.cmudict = temp_dict
                print(f"AlternativeCMUDict: Successfully loaded {len(self.cmudict)} entries from {location}")

        except FileNotFoundError:
            print(f"ERROR: AlternativeCMUDict - Dictionary file not found at {location}")
            # Keep self.cmudict as {}
        except Exception as e:
            print(f"ERROR: AlternativeCMUDict - Error loading dictionary from {location}: {e}")
            traceback.print_exc() # Print full traceback
            # Keep self.cmudict as {}


    def get_arpabet(self,text):
        """
        :param text: accepts a single word
        :return: formatted phoneme string or original text if OOV
        """
        self.n += 1 # Increment lookup count regardless of OOV

        # Normalize input text to uppercase for lookup
        text_upper = text.upper()

        # Apply Dutch OOV rules if configured
        # Use getattr for safe access to config attributes
        if getattr(getattr(self.conf, 'phoneme', None), 'dutch_oov_rule', False):
            # Use a dictionary for replacements - more efficient and readable
            replacements = {
                "Ë": "&euml;", "É": "&eacute;", "À": "&agrave;", "Ê": "&ecirc;",
                "Ï": "&iuml;", "È": "&egrave;", "Ü": "&uuml;", "Ç": "&ccedil;",
                "Ö": "&ouml;", "Ä": "&auml;"
            }
            for char, entity in replacements.items():
                # Apply replacements before lookup
                text_upper = text_upper.replace(char, entity) # Replace uppercase char with entity

        # Perform dictionary lookup
        if text_upper in self.cmudict:
            pronunciation = self.cmudict[text_upper]
            if pronunciation: # Check if pronunciation is not empty
                # Format with curly braces only if needed by downstream processing
                # If your phoneme analysis expects simple space-separated phonemes, don't add braces
                # return pronunciation # Example: return space-separated phonemes directly
                # Or keep the original formatting if required:
                return " ".join(["{" + phoneme + "}" for phoneme in pronunciation.split()])
            else:
                # Word is in dict but has no pronunciation, treat as OOV
                self.oov += 1
                return text # Return original text
        else:
            # Word not found in dictionary
            # print(f"OOV: {text}") # Optional debug print
            self.oov += 1
            return text # Return original text



class WERDetails:
    def __init__(self, location: str, skip_calculation=False, config=None):
        print(f"\n--- WERDetails INIT Start for: {location} ---") # DEBUG
        self.config = config
        self.location = location
        self.skip_calculation = skip_calculation
        self.data = None # Initialize self.data to None
        self.cmudict = {} # Initialize cmudict as empty dict

        # --- Dictionary Loading ---
        print("DEBUG: Starting dictionary loading...") # DEBUG
        try:
            # Check if config exists and alternative_cmudict is True
            if config and getattr(getattr(config, 'phoneme', None), 'alternative_cmudict', False):
                dict_path = getattr(config.phoneme, 'dictionary', None)
                if dict_path:
                    # Resolve relative path based on config file location if needed
                    if not os.path.isabs(dict_path) and hasattr(config, 'config_path'):
                         config_dir = os.path.dirname(config.config_path)
                         dict_path = os.path.join(config_dir, dict_path)
                         print(f"DEBUG: Resolved relative dictionary path to: {dict_path}") # DEBUG
                    else:
                         print(f"DEBUG: Using dictionary path as is: {dict_path}") # DEBUG

                    if not os.path.exists(dict_path):
                         print(f"ERROR: Dictionary file not found at: {dict_path}")
                         # self.cmudict remains {}
                    else:
                         # Initialize AlternativeCMUDict
                         self.cmudict = AlternativeCMUDict(dict_path, config)
                         # Check if AlternativeCMUDict loaded successfully (check internal dict)
                         if hasattr(self.cmudict, 'cmudict') and self.cmudict.cmudict:
                              print(f"DEBUG: Initialized AlternativeCMUDict. Size: {len(self.cmudict.cmudict)}") # DEBUG
                         else:
                              print("DEBUG: AlternativeCMUDict initialization failed or loaded empty dict.") # DEBUG
                              self.cmudict = {} # Ensure it's an empty dict on failure
                else:
                    print("ERROR: alternative_cmudict is True but config.phoneme.dictionary path is missing.")
            else:
                # Handle case where AlternativeCMUDict is not used (e.g., default dict or no dict)
                print("DEBUG: AlternativeCMUDict not configured or config missing. No dictionary loaded by default.")
                # self.cmudict = getattr(text, '_cmudict', {}) # Example if using 'text' module
                self.cmudict = {} # Default to empty dict

            print("DEBUG: Dictionary loading section finished.") # DEBUG
        except ImportError as e:
             print(f"ERROR during dictionary loading (ImportError): {e}") # DEBUG
             self.cmudict = {} # Ensure cmudict exists even on error
        except FileNotFoundError as e:
             print(f"ERROR during dictionary loading (FileNotFound): {e}") # DEBUG
             self.cmudict = {}
        except Exception as e:
            print(f"ERROR during dictionary loading (General Exception): {e}") # DEBUG
            traceback.print_exc() # Print full traceback for unexpected errors
            self.cmudict = {} # Ensure cmudict exists even on error


        # --- Sentence Loading & Alignment ---
        if not skip_calculation:
            ref_sentences, hyp_sentences = None, None # Initialize
            try:
                print(f"DEBUG: Attempting to load sentences from {location}...") # DEBUG
                # Ensure load_sentences exists and handles potential errors
                if hasattr(self, 'load_sentences'):
                     ref_sentences, hyp_sentences = self.load_sentences(asr=getattr(self.config, 'asr', 'kaldi'))
                     if ref_sentences is not None and hyp_sentences is not None:
                          print(f"DEBUG: Successfully loaded {len(ref_sentences)} sentences.") # DEBUG
                     else:
                          print("DEBUG: Sentence loading returned None or empty lists.") # DEBUG
                else:
                     print("ERROR: load_sentences method not found in WERDetails.")

            except FileNotFoundError as e:
                 print(f"ERROR during sentence loading (FileNotFound): {e}") # DEBUG
                 ref_sentences, hyp_sentences = None, None
            except Exception as e:
                 print(f"ERROR during sentence loading (General Exception): {e}") # DEBUG
                 traceback.print_exc()
                 ref_sentences, hyp_sentences = None, None

            # Proceed only if sentences were loaded successfully and are not empty
            if ref_sentences and hyp_sentences: # Check if lists are not None AND not empty
                try:
                    print("DEBUG: Attempting phoneme alignment...") # DEBUG
                    # Ensure phoneme_alignment exists and self.cmudict is valid
                    if not hasattr(self, 'phoneme_alignment'):
                         print("ERROR: phoneme_alignment method not found.")
                         self.data = None
                    # Check if cmudict is an instance of the class or a dict, and if it's populated
                    elif not isinstance(self.cmudict, (dict, AlternativeCMUDict)) or \
                         (isinstance(self.cmudict, AlternativeCMUDict) and not self.cmudict.cmudict) or \
                         (isinstance(self.cmudict, dict) and not self.cmudict):
                         print("ERROR: Dictionary (self.cmudict) is not loaded correctly or is empty for alignment.")
                         self.data = None
                    else:
                         # Pass sentences to phoneme_alignment
                         self.all_ref_phonemes, self.all_hyp_phonemes, self.manipulations = self.phoneme_alignment(ref_sentences, hyp_sentences)
                         # Check if alignment produced results
                         if hasattr(self, 'all_ref_phonemes') and self.all_ref_phonemes:
                              self.data = True # Indicate success ONLY if alignment finishes and produces data
                              print(f"DEBUG: Phoneme alignment successful. Found {len(self.all_ref_phonemes)} aligned phonemes total.") # DEBUG
                         else:
                              print("ERROR: Phoneme alignment finished but produced no phonemes.")
                              self.data = None

                except Exception as e:
                    print(f"ERROR during phoneme alignment: {e}") # DEBUG
                    traceback.print_exc() # Print full traceback
                    self.data = None # Ensure data is None on failure
            else:
                 if ref_sentences is None or hyp_sentences is None:
                     print(f"DEBUG: Skipping phoneme alignment due to sentence loading failure.") # DEBUG
                 else: # Sentences were loaded but were empty lists
                     print(f"DEBUG: Skipping phoneme alignment because no sentences were loaded (empty lists).") # DEBUG
                 self.data = None # Ensure data is None if alignment skipped
        else:
            print("DEBUG: Skipping sentence loading and alignment (skip_calculation=True).") # DEBUG
            self.data = None # Set data to None if skipping calculation


        # --- OOV Calculation Block ---
        # Check if alternative_cmudict is True AND if cmudict object exists and has 'n' > 0
        # Also check if alignment succeeded (self.data is True)
        if config and getattr(getattr(config, 'phoneme', None), 'alternative_cmudict', False) \
           and isinstance(self.cmudict, AlternativeCMUDict) \
           and hasattr(self.cmudict, 'n') and hasattr(self.cmudict, 'oov') \
           and self.data is True: # Check if data processing was successful
            if self.cmudict.n > 0:
                # Calculate and print OOV rate
                oov_rate = (self.cmudict.oov / self.cmudict.n) * 100
                print(f"OOV rate: {oov_rate:.2f}% ({self.cmudict.oov}/{self.cmudict.n})")
            else:
                # This case indicates no words were processed via get_arpabet
                print(f"DEBUG: No words processed through dictionary lookup for {location}. Cannot calculate OOV rate.")
        # else:
            # Optional: Log why OOV wasn't calculated
            # if not (config and getattr(getattr(config, 'phoneme', None), 'alternative_cmudict', False)):
            #     print("DEBUG: OOV calculation skipped (alternative_cmudict not True).")
            # elif not isinstance(self.cmudict, AlternativeCMUDict):
            #     print("DEBUG: OOV calculation skipped (cmudict not AlternativeCMUDict instance).")
            # elif self.data is not True:
            #     print("DEBUG: OOV calculation skipped (data processing failed).")
            # elif not (hasattr(self.cmudict, 'n') and hasattr(self.cmudict, 'oov')):
            #     print("DEBUG: OOV calculation skipped (cmudict missing n/oov attributes).")
        # --- End OOV Calculation Block ---


        # --- Final Check ---
        if not skip_calculation and self.data is None:
             print(f"DEBUG: WARNING - Initialization failed or was skipped for {location}. self.data is None.") # DEBUG

        print(f"--- WERDetails INIT End for: {location} ---") # DEBUG


    # --- phoneme_alignment method ---
    def phoneme_alignment(self, ref_sentences, hyp_sentences):
        """
        Aligns reference and hypothesis phonemes for given sentences.
        """
        print("DEBUG: Inside phoneme_alignment method.") # DEBUG
        all_ref_phonemes = list()
        all_hyp_phonemes = list()
        all_manipulations = list()

        # Realign everything on the phoneme-level
        num_sentences = len(ref_sentences)
        for i, (ref_sentence, hyp_sentence) in enumerate(zip(ref_sentences, hyp_sentences)):
            # print(f"DEBUG: Aligning sentence {i+1}/{num_sentences}") # Optional: progress per sentence
            try:
                clean_ref_sentence = self.clean_non_words(ref_sentence)
                clean_hyp_sentence = self.clean_non_words(hyp_sentence)

                # Convert words to phonemes
                # Ensure words_to_phoneme exists and handles errors
                if hasattr(self, 'words_to_phoneme'):
                    ref_phoneme_list = self.words_to_phoneme(clean_ref_sentence, stress_cleaned=True)
                    hyp_phoneme_list = self.words_to_phoneme(clean_hyp_sentence, stress_cleaned=True)
                else:
                    print(f"ERROR: words_to_phoneme method not found. Skipping sentence {i+1}.")
                    continue # Skip this sentence pair

                # Align phoneme sequences
                # Ensure align_sequences exists and handles errors
                if 'align_sequences' in globals():
                     alignments, manipulations = align_sequences(ref_phoneme_list, hyp_phoneme_list)
                     reference_aligned, hypothesis_aligned = alignments
                else:
                     print(f"ERROR: align_sequences function not found. Skipping sentence {i+1}.")
                     continue # Skip this sentence pair

                # Basic check on alignment output
                if len(reference_aligned) != len(hypothesis_aligned) or len(reference_aligned) != len(manipulations):
                     print(f"ERROR: Alignment output length mismatch for sentence {i+1}. Skipping.")
                     continue # Skip this sentence pair

                # Extend lists
                all_ref_phonemes.extend(reference_aligned)
                all_hyp_phonemes.extend(hypothesis_aligned)
                all_manipulations.extend(manipulations)

            except Exception as e:
                 print(f"ERROR processing sentence pair {i+1}: {e}")
                 traceback.print_exc()
                 # Decide whether to skip this sentence or stop entirely
                 continue # Skip to next sentence pair

        # Add missing phonemes (ensure sets are created from potentially non-empty lists)
        if all_ref_phonemes or all_hyp_phonemes:
            hyp_set = set(all_hyp_phonemes)
            ref_set = set(all_ref_phonemes)
            missing_phonemes = hyp_set.symmetric_difference(ref_set)

            for phoneme in missing_phonemes:
                all_ref_phonemes.append(phoneme)
                all_hyp_phonemes.append(phoneme)
                all_manipulations.append("e") # e is correct

            # Sanity check
            if set(all_ref_phonemes) != set(all_hyp_phonemes):
                 print("WARNING: Phoneme sets still differ after adding missing ones.")
        else:
             print("DEBUG: No phonemes generated during alignment.")


        print(f"DEBUG: phoneme_alignment finished. Total aligned items: {len(all_ref_phonemes)}") # DEBUG
        return all_ref_phonemes, all_hyp_phonemes, all_manipulations

    # --- load_sentences method ---
    def load_sentences(self, asr="kaldi") -> Tuple[list,list]:
        """
        Loads the references and hypothesis sentences from the WER details file
        :return: two list of sentences
        """
        print(f"DEBUG: Inside load_sentences for {self.location} (asr={asr})") # DEBUG
        if not os.path.exists(self.location):
             # Raise FileNotFoundError explicitly if file doesn't exist
             raise FileNotFoundError(f"WER details file not found at {self.location}")

        ref_words_sentencewise = []
        hyp_words_sentencewise = []

        with open(self.location, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        # Determine line format based on asr type
        ref_prefix = "ref" if asr == "kaldi" else "REF:"
        hyp_prefix = "hyp" if asr == "kaldi" else "HYP:"
        start_index = 2 if asr == "kaldi" else 1 # Index where words start

        for line_num, line in enumerate(all_lines, 1):
             parts = line.split() # Split by whitespace
             if not parts: continue # Skip empty lines

             line_type = parts[1] if asr == "kaldi" and len(parts) > 1 else (parts[0] if asr != "kaldi" else None)

             if line_type == ref_prefix:
                  # Check if there are words beyond the prefix and ID
                  if len(parts) > start_index:
                       ref_words_sentencewise.append(parts[start_index:])
                  else:
                       ref_words_sentencewise.append([]) # Append empty list if no words
             elif line_type == hyp_prefix:
                  if len(parts) > start_index:
                       hyp_words_sentencewise.append(parts[start_index:])
                  else:
                       hyp_words_sentencewise.append([]) # Append empty list if no words

        # Basic validation
        if len(ref_words_sentencewise) != len(hyp_words_sentencewise):
             print(f"Warning: Mismatch in number of ref ({len(ref_words_sentencewise)}) and hyp ({len(hyp_words_sentencewise)}) lines found in {self.location}")
             # Decide how to handle: return None, raise error, or try to pair anyway?
             # Returning None might be safest to signal failure.
             return None, None

        print(f"DEBUG: load_sentences finished for {self.location}. Loaded {len(ref_words_sentencewise)} pairs.") # DEBUG
        return ref_words_sentencewise, hyp_words_sentencewise

    # --- Other methods (raw_confusion_matrix, all_pers, etc.) ---
    # Add checks within these methods to ensure self.data is True
    # and self.all_ref_phonemes etc. exist before using them.

    def raw_confusion_matrix(self) -> Tuple[np.ndarray, list]:
        """Calculates the phoneme confusion matrix"""
        if getattr(self, 'data', None) is not True or not hasattr(self, 'all_ref_phonemes'):
            print("Error: Cannot calculate confusion matrix. Data not processed or phonemes missing.")
            return np.array([]), [] # Return empty results
        labels = sorted(list(set(self.all_ref_phonemes)))
        conf_matrix = confusion_matrix(self.all_ref_phonemes, self.all_hyp_phonemes, labels=labels) # Pass labels explicitly
        return conf_matrix, labels

    def all_pers(self) -> Tuple[list, List[float]]:
        """Returns the PER for each phoneme"""
        if getattr(self, 'data', None) is not True or not hasattr(self, 'all_ref_phonemes'):
            print("Error: Cannot calculate PERs. Data not processed or phonemes missing.")
            return [], [] # Return empty results
        labels = sorted(list(set(self.all_ref_phonemes)))
        pers = []
        for phoneme in labels:
             try:
                  pers.append(self.per_per_phoneme(phoneme))
             except Exception as e:
                  print(f"Error calculating PER for phoneme '{phoneme}': {e}")
                  pers.append(np.nan) # Append NaN or handle error differently
        return labels, pers

    def phoneme_count(self) -> Tuple[List[str], List[int]]:
        """Returns the number of phonemes in the reference sentences, per phoneme"""
        if getattr(self, 'data', None) is not True or not hasattr(self, 'all_ref_phonemes'):
            print("Error: Cannot calculate phoneme count. Data not processed or phonemes missing.")
            return [], [] # Return empty results
        # np.unique guarantees sorted return
        labels, counts = np.unique(self.all_ref_phonemes, return_counts=True)
        return list(labels), list(counts)

    # --- Static methods and helper methods (keep as is, ensure they are robust) ---
    @staticmethod
    def clean_non_words(sentence: list) -> list:
        """Clears the WERDetails segments from the *** s"""
        # Ensure input is a list
        if not isinstance(sentence, list):
             print(f"Warning: clean_non_words expected list, got {type(sentence)}. Returning empty list.")
             return []
        words = list(filter(lambda a: a != "***", sentence))
        return words

    @staticmethod
    def arpabet_cleaner(arpabet: str, stress_remove: bool = False) -> List[str]:
        """Cleans ARPAbet string"""
        if not isinstance(arpabet, str):
             # print(f"Warning: arpabet_cleaner expected string, got {type(arpabet)}. Returning empty list.")
             return []
        arpabet_wo_braces = arpabet.replace("{", "").replace("}", "")
        arpabet_split = arpabet_wo_braces.split()
        if stress_remove:
            arpabet_split = [p.rstrip('0123456789') for p in arpabet_split if isinstance(p, str)]
        return arpabet_split

    def word_to_phoneme(self, word: str, stress_cleaned: bool) -> List[str]:
        """Converts a single word to phonemes using the configured dictionary method."""
        # Ensure word is a string
        if not isinstance(word, str):
             # print(f"Warning: word_to_phoneme expected string, got {type(word)}. Returning empty list.")
             return []

        # Use getattr for safe access to config and cmudict
        use_alt_dict = getattr(getattr(self.config, 'phoneme', None), 'alternative_cmudict', False)
        cmudict_obj = getattr(self, 'cmudict', None)

        if use_alt_dict:
            if isinstance(cmudict_obj, AlternativeCMUDict) and hasattr(cmudict_obj, 'get_arpabet'):
                 arpabet_str = cmudict_obj.get_arpabet(word)
                 # Check if the result indicates OOV (returns original word)
                 if arpabet_str == word:
                      # print(f"DEBUG: OOV word '{word}' using AlternativeCMUDict.")
                      return [] # Return empty list for OOV words
                 else:
                      # Clean the pronunciation string (assuming it needs cleaning)
                      return self.arpabet_cleaner(arpabet_str, False) # Pass False for stress_cleaned with alt dict
            else:
                 print(f"Warning: AlternativeCMUDict configured but not loaded correctly or missing get_arpabet method.")
                 return [] # Return empty list if dict not usable
        else:
             # --- Original logic using 'text' module (if applicable) ---
             # This part requires the 'text' module and its functions
             # try:
             #     # Ensure text module and get_arpabet exist
             #     if 'text' in sys.modules and hasattr(sys.modules['text'], 'get_arpabet'):
             #         # Ensure cmudict_obj is a dict if required by text.get_arpabet
             #         if not isinstance(cmudict_obj, dict): cmudict_obj = {}
             #         arpabet_str = sys.modules['text'].get_arpabet(word, cmudict_obj)
             #         # Check if OOV (assuming text.get_arpabet returns original word on OOV)
             #         if arpabet_str == word:
             #              return []
             #         else:
             #              return self.arpabet_cleaner(arpabet_str, stress_cleaned)
             #     else:
             #         print("Warning: 'text' module or 'get_arpabet' function not available.")
             #         return []
             # except Exception as e:
             #     print(f"Error during default dictionary lookup for '{word}': {e}")
             #     return []
             # --- End original logic ---
             # If not using 'text' module, return empty for non-alt-dict case
             print("Warning: Dictionary lookup method unclear (not alternative_cmudict, 'text' module logic commented out).")
             return []


    def words_to_phoneme(self, words: list, stress_cleaned: bool) -> List[str]:
        """Converts a list of words to a flat list of phonemes."""
        if not isinstance(words, list):
             print(f"Warning: words_to_phoneme expected list, got {type(words)}. Returning empty list.")
             return []
        phoneme_list = []
        for word in words:
             # word_in_dict check might be redundant if word_to_phoneme handles OOV by returning []
             # if self.word_in_cmu_dict(word): # Check if word is in dict before converting
             phonemes = self.word_to_phoneme(word, stress_cleaned)
             if phonemes: # Only extend if conversion was successful (not OOV)
                  phoneme_list.extend(phonemes)
        return phoneme_list

    def word_in_cmu_dict(self, word) -> bool:
        """Checks if a word exists in the configured dictionary."""
        # This check might be less reliable depending on how get_arpabet handles OOV
        # It's often better to just try the conversion and see if it returns an empty list.
        use_alt_dict = getattr(getattr(self.config, 'phoneme', None), 'alternative_cmudict', False)
        cmudict_obj = getattr(self, 'cmudict', None)

        if use_alt_dict:
            if isinstance(cmudict_obj, AlternativeCMUDict) and hasattr(cmudict_obj, 'get_arpabet'):
                 decoded = cmudict_obj.get_arpabet(word)
                 # Check if the result is different from the original word (indicates found)
                 # Or if the result contains '{' (original logic)
                 return decoded != word # or "{" in decoded
            else:
                 return False # Cannot check if dict not loaded
        else:
             # --- Original logic using 'text' module ---
             # try:
             #     if 'text' in sys.modules and hasattr(sys.modules['text'], 'get_arpabet'):
             #         if not isinstance(cmudict_obj, dict): cmudict_obj = {}
             #         decoded = sys.modules['text'].get_arpabet(word, cmudict_obj)
             #         return decoded != word # or "{" in decoded
             #     else:
             #         return False
             # except Exception:
             #     return False
             # --- End original logic ---
             return False # Default if no method available

    # --- POA/MOA methods (Keep as is, but ensure self.converter_table is loaded if needed) ---
    # These methods rely on self.converter_table which doesn't seem to be loaded in __init__
    # You might need to load it from config.phoneme.conversion_mapping similar to figure_generator_2.py
    # Add error handling if self.converter_table is not available.

    def _load_converter_table(self):
         """Helper to load the phoneme conversion table."""
         if hasattr(self, 'converter_table') and self.converter_table is not None:
              return # Already loaded

         self.converter_table = None # Initialize
         mapping_path = getattr(getattr(self.config, 'phoneme', None), 'conversion_mapping', None)
         phoneme_col_name = getattr(getattr(self.config, 'phoneme', None), 'phoneme_name', None) # Get phoneme column name from config

         if mapping_path and phoneme_col_name:
              # Resolve path relative to config if necessary
              if not os.path.isabs(mapping_path) and hasattr(self.config, 'config_path'):
                   config_dir = os.path.dirname(self.config.config_path)
                   mapping_path = os.path.join(config_dir, mapping_path)

              if os.path.exists(mapping_path):
                   try:
                        df = pd.read_csv(mapping_path)
                        df = df[df[phoneme_col_name].notna()] # Use configured phoneme column name
                        df = df.fillna(0)
                        # Set index using the configured phoneme column name
                        self.converter_table = df.set_index(df[phoneme_col_name])
                        print(f"DEBUG: Loaded converter table from {mapping_path}")
                   except Exception as e:
                        print(f"ERROR loading converter table from {mapping_path}: {e}")
              else:
                   print(f"ERROR: Converter table file not found: {mapping_path}")
         else:
              print("ERROR: conversion_mapping or phoneme_name not defined in config. Cannot load converter table.")


    def phoneme_to_poa(self, phoneme: str) -> str:
        self._load_converter_table() # Ensure table is loaded
        if self.converter_table is None: return "ERROR" # Handle missing table

        if not isinstance(phoneme, str): return "INVALID_INPUT"
        if (phoneme == " ") or (phoneme == "[SPN]"): return phoneme # Handle space/SPN

        poas = getattr(getattr(self.config, 'phoneme', None), 'poa', [])
        if not poas: return "NO_POA_CONFIG"

        try:
            # Check if phoneme exists in the index
            if phoneme not in self.converter_table.index:
                 # print(f"Warning: Phoneme '{phoneme}' not found in converter table index.")
                 return "UNKNOWN_PHONEME"

            # Ensure requested POAs exist as columns
            valid_poa_cols = [p for p in poas if p in self.converter_table.columns]
            if not valid_poa_cols:
                 print(f"Warning: None of the configured POAs {poas} found in converter table columns.")
                 return "NO_POA_COLS"

            # Get the row for the phoneme and select only valid POA columns
            phoneme_row = self.converter_table.loc[phoneme, valid_poa_cols]

            # Find the POA column with the maximum value (usually 1)
            poa_idx = phoneme_row.idxmax() # Returns the column name (POA) with the max value
            return poa_idx
        except KeyError:
             # This might happen if phoneme is in index but not valid_poa_cols somehow
             print(f"Warning: KeyError looking up POA for phoneme '{phoneme}'.")
             return "LOOKUP_ERROR"
        except Exception as e:
            print(f"Error in phoneme_to_poa for '{phoneme}': {e}")
            return "ERROR"


    def phoneme_to_moa(self, phoneme: str) -> str:
        self._load_converter_table() # Ensure table is loaded
        if self.converter_table is None: return "ERROR" # Handle missing table

        if not isinstance(phoneme, str): return "INVALID_INPUT"
        if (phoneme == " ") or (phoneme == "[SPN]"): return phoneme # Handle space/SPN

        moas = getattr(getattr(self.config, 'phoneme', None), 'moa', [])
        if not moas: return "NO_MOA_CONFIG"

        try:
            # Check if phoneme exists in the index
            if phoneme not in self.converter_table.index:
                 # print(f"Warning: Phoneme '{phoneme}' not found in converter table index.")
                 return "UNKNOWN_PHONEME"

            # Ensure requested MOAs exist as columns
            valid_moa_cols = [m for m in moas if m in self.converter_table.columns]
            if not valid_moa_cols:
                 print(f"Warning: None of the configured MOAs {moas} found in converter table columns.")
                 return "NO_MOA_COLS"

            # Get the row for the phoneme and select only valid MOA columns
            phoneme_row = self.converter_table.loc[phoneme, valid_moa_cols]

            # Find the MOA column with the maximum value (usually 1)
            moa_idx = phoneme_row.idxmax() # Returns the column name (MOA) with the max value
            return moa_idx
        except KeyError:
             print(f"Warning: KeyError looking up MOA for phoneme '{phoneme}'.")
             return "LOOKUP_ERROR"
        except Exception as e:
            print(f"Error in phoneme_to_moa for '{phoneme}': {e}")
            return "ERROR"


    def poa_to_phonemes(self, poa: str) -> list:
        """Converts a place of articulation feature to the corresponding list of phonemes"""
        self._load_converter_table() # Ensure table is loaded
        if self.converter_table is None: return []

        poa_list_config = getattr(getattr(self.config, 'phoneme', None), 'poa', [])
        if poa not in poa_list_config:
             print(f"Warning: POA '{poa}' not found in config.phoneme.poa list.")
             return []
        if poa not in self.converter_table.columns:
             print(f"Warning: POA '{poa}' not found as a column in the converter table.")
             return []

        try:
            poa_filter = self.converter_table.loc[:, poa]
            # Get index values where the filter is 1 (or max value if not strictly 0/1)
            phonemes = poa_filter.index[poa_filter == poa_filter.max()].tolist()

            if not phonemes:
                print(f"Warning: No phonemes found for POA '{poa}' in the converter table.")
            return phonemes
        except Exception as e:
             print(f"Error in poa_to_phonemes for '{poa}': {e}")
             return []


    def moa_to_phonemes(self, moa: str) -> list:
        """Converts a manner of articulation feature to the corresponding list of phonemes"""
        self._load_converter_table() # Ensure table is loaded
        if self.converter_table is None: return []

        moa_list_config = getattr(getattr(self.config, 'phoneme', None), 'moa', [])
        if moa not in moa_list_config:
             print(f"Warning: MOA '{moa}' not found in config.phoneme.moa list.")
             return []
        if moa not in self.converter_table.columns:
             print(f"Warning: MOA '{moa}' not found as a column in the converter table.")
             return []

        try:
            moa_filter = self.converter_table.loc[:, moa]
            # Get index values where the filter is 1 (or max value)
            phonemes = moa_filter.index[moa_filter == moa_filter.max()].tolist()

            if not phonemes:
                 print(f"Warning: No phonemes found for MOA '{moa}' in the converter table.")
            return phonemes
        except Exception as e:
             print(f"Error in moa_to_phonemes for '{moa}': {e}")
             return []


    # --- Methods requiring successful alignment (add checks) ---
    def per_per_phoneme(self, phoneme: str) -> float:
        """Returns per for a given phoneme"""
        if getattr(self, 'data', None) is not True or not hasattr(self, 'all_ref_phonemes'):
             print(f"Error: Cannot calculate PER for '{phoneme}'. Data not processed or phonemes missing.")
             return np.nan # Return NaN for error
        if phoneme not in set(self.all_ref_phonemes):
             # print(f"Warning: Phoneme '{phoneme}' not found in reference phonemes. Returning NaN for PER.")
             return np.nan # Return NaN if phoneme not present

        try:
            per = per_phoneme_per(phoneme, self.all_ref_phonemes, self.all_hyp_phonemes, self.manipulations)
            return per
        except Exception as e:
             print(f"Error calculating PER for phoneme '{phoneme}' using per_utils: {e}")
             return np.nan


    def afer_per_phoneme(self, phonemes: list) -> float:
        """Calculates AFER for a list of phonemes (representing a feature)."""
        if getattr(self, 'data', None) is not True or not hasattr(self, 'all_ref_phonemes'):
             print(f"Error: Cannot calculate AFER for {phonemes}. Data not processed or phonemes missing.")
             return np.nan
        if not isinstance(phonemes, list):
             print(f"Error: afer_per_phoneme expects a list of phonemes, got {type(phonemes)}.")
             return np.nan

        try:
            # Ensure afer function exists and handles potential errors (like division by zero)
            if 'afer' in globals():
                 afer_val = afer(phonemes, self.all_ref_phonemes, self.all_hyp_phonemes, self.manipulations)
                 return afer_val
            else:
                 print("Error: 'afer' function not found.")
                 return np.nan
        except Exception as e:
             # Catch potential errors from afer function (e.g., "Not in hypothesis" if n=0)
             print(f"Error calculating AFER for feature {phonemes} using per_utils: {e}")
             return np.nan


    def all_moa_afers(self) -> Tuple[list, List[float]]:
        """Calculate Manner of Articulation Articulatory Feature Error Rate"""
        if getattr(self, 'data', None) is not True:
             print("Error: Cannot calculate MOA AFERs. Data not processed.")
             return [], []
        moas = getattr(getattr(self.config, 'phoneme', None), 'moa', [])
        if not moas:
             print("Error: config.phoneme.moa not defined.")
             return [], []

        afers = []
        valid_moas = []
        for moa in moas:
             phonemes_for_moa = self.moa_to_phonemes(moa)
             if phonemes_for_moa: # Only calculate if phonemes were found for this MOA
                  afer_val = self.afer_per_phoneme(phonemes_for_moa)
                  afers.append(afer_val)
                  valid_moas.append(moa)
             else:
                  print(f"Skipping AFER calculation for MOA '{moa}' as no corresponding phonemes were found.")
                  # Optionally append NaN or skip the MOA entirely
                  # afers.append(np.nan)
                  # valid_moas.append(moa)

        return valid_moas, afers


    def all_poa_afers(self) -> Tuple[list, List[float]]:
        """Calculate Place of Articulation Articulatory Feature Error Rate"""
        if getattr(self, 'data', None) is not True:
             print("Error: Cannot calculate POA AFERs. Data not processed.")
             return [], []
        poas = getattr(getattr(self.config, 'phoneme', None), 'poa', [])
        if not poas:
             print("Error: config.phoneme.poa not defined.")
             return [], []

        afers = []
        valid_poas = []
        for poa in poas:
             phonemes_for_poa = self.poa_to_phonemes(poa)
             if phonemes_for_poa: # Only calculate if phonemes were found for this POA
                  afer_val = self.afer_per_phoneme(phonemes_for_poa)
                  afers.append(afer_val)
                  valid_poas.append(poa)
             else:
                  print(f"Skipping AFER calculation for POA '{poa}' as no corresponding phonemes were found.")

        return valid_poas, afers


# --- Main execution block (for testing corpus.py directly) ---
if __name__ == '__main__':
    print("Running corpus.py directly (for testing purposes)...")

    # Example usage (replace with actual paths and config)
    # You need a sample config file and a sample per_utt file to test this
    test_config_path = 'configs/dutch.yaml' # Adjust path
    test_per_utt_path = 'experiments/jasmin_example/scoring_kaldi/wer_details/per_utt' # Adjust path

    if os.path.exists(test_config_path) and os.path.exists(test_per_utt_path):
        print(f"\n--- Testing WERDetails with {test_per_utt_path} ---")
        try:
            test_config = HParam(test_config_path)
            test_config.config_path = test_config_path # Add config path attribute

            # Test initialization
            details = WERDetails(test_per_utt_path, skip_calculation=False, config=test_config)

            # Test methods if initialization was successful
            if getattr(details, 'data', None) is True:
                print("\n--- Testing WERDetails methods ---")
                labels, pers = details.all_pers()
                print(f"PERs calculated for {len(labels)} phonemes.")
                # print(list(zip(labels, pers)))

                counts_labels, counts = details.phoneme_count()
                print(f"Counts calculated for {len(counts_labels)} phonemes.")
                # print(list(zip(counts_labels, counts)))

                # Add tests for other methods like confusion matrices, AFERs if needed
                # moa_labels, moa_afers = details.all_moa_afers()
                # print(f"MOA AFERs: {list(zip(moa_labels, moa_afers))}")

            else:
                print("\nSkipping method tests as WERDetails initialization failed.")

        except Exception as e:
            print(f"\nError during corpus.py self-test: {e}")
            traceback.print_exc()
    else:
        print("\nSkipping self-test: Config or per_utt file not found at specified paths.")
        print(f"Config path checked: {os.path.abspath(test_config_path)}")
        print(f"per_utt path checked: {os.path.abspath(test_per_utt_path)}")

