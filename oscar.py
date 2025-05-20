import re
import os
import sys
import subprocess
import string # Import string for punctuation check
import collections # Import collections for Counter
from datasets import load_dataset
from tqdm import tqdm
# Optional: from langdetect import detect, LangDetectException # Uncomment if using language detection

# --- Configuration ---

# Phoneme Targets (VERIFY/ADJUST THESE IPA SYMBOLS)
# Based on large-v3-turbo Importance Scores for focused analysis

# Phonemes where the NL group had a notably higher PER than the FR group
# (selected based on the most negative Importance_Score, indicating NL was worse)
TARGET_PHONEMES_NL = ['j', 'd', 'f', 'a']

# Phonemes where the FR group had a notably higher PER than the NL group
# (selected based on the highest positive Importance_Score, indicating FR was worse)
# Using IPA representations that should be compatible with espeak-ng output
# and the existing get_phonemes regex.
TARGET_PHONEMES_FR = ['ŋ', 'y', 'b', 's']


# Output Files
OUTPUT_SENTENCES_NL_RICH = 'output_nl_rich_sentences_turbo.txt'
OUTPUT_SENTENCES_FR_RICH = 'output_fr_rich_sentences_turbo.txt'

# Density Thresholds (Adjust as needed)
DENSITY_THRESHOLD_NL = 0.05 # Example: at least 5% of phonemes in sentence are one of the NL targets
DENSITY_THRESHOLD_FR = 0.05 # Example: at least 5% of phonemes in sentence are one of the FR targets (adjust if too low/high)

# Sentence Length Constraints
MIN_WORDS = 5
MAX_WORDS = 25

# Cleaning Parameters (Adjust as needed)
MIN_CHARS = 25 # Increased slightly
MAX_UPPER_RATIO = 0.25 # Stricter uppercase ratio
MIN_ALPHA_RATIO = 0.65 # Require slightly more alphabetic chars
MAX_DIGIT_RATIO = 0.15 # Max 15% digits allowed

# Processing Control
PROCESS_LIMIT = None # Set to a number (e.g., 500000) for testing, or None to run fully
SAVE_INTERVAL = 1000 # How often to save results

# Dataset Details
OSCAR_DATASET_NAME = "oscar-corpus/OSCAR-2201"
OSCAR_LANGUAGE = "nl"

# --- Keywords/Patterns to Filter Out ---
BAD_KEYWORDS = [
    # Web/UI
    'javascript', 'cookie', 'browser', 'copyright', 'http:', 'https:', '.nl', '.com', '.org', '.net', '.io', '.dev', '.app',
    'klik hier', 'lees meer', 'login', 'logout', 'password', 'gebruikersnaam', 'account', 'profiel', 'instellingen',
    'search', 'filter', 'submit', 'download', 'upload', 'privacy', 'terms', 'disclaimer', 'subscribe', 'unsubscribe',
    'newsletter', 'help', 'faq', 'contact', 'about us', 'home', 'next', 'previous', 'page', 'item', 'cart', 'checkout',
    'register', 'username', 'email address', 'share', 'tweet', 'like', 'follow', 'comment', 'reply',
    # Technical/Code/Markup
    'html', 'css', 'php', 'json', 'xml', 'api', 'sdk', 'url', 'uri', 'parameter', 'variable', 'function', 'class',
    'object', 'array', 'null', 'undefined', 'true', 'false', 'error code', 'status code', 'warning:', 'notice:', 'deprecated',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar', '.jpg', '.png', '.gif', '.svg', '.mp3', '.mp4', '.avi', '.log', '.yaml',
    # Placeholders/Metadata
    'lorem ipsum', 'placeholder', 'example', 'template', 'default', 'untitled', 'unknown', 'version', 'revision',
    'author', 'date', 'timestamp', 'id:', 'uuid', 'isbn', 'tags:', 'categorie:',
    # Ads/Tracking
    'ad', 'advertisement', 'sponsored', 'tracking', 'analytics',
    # Finance/Commerce
    'haccp', 'btw', '€', '$', 'usd', 'eur', 'price', 'discount', 'sale', 'order', 'shipping',
    # Common Warnings/Info
    'error', 'helaas', 'ondersteund', 'bericht', 'nieuws', 'update',
    # Licensing
    'cc by', 'cc-by', 'license', 'licentie', 'rights reserved',
    # Other common noise
    'menu', 'ingredienten', 'gerecht', 'foto', 'afbeelding', 'description:',
]

BAD_PATTERNS = re.compile(
    r'https?://\S+|'           # URLs
    r'www\.\S+|'              # URLs starting with www
    r'\S+@\S+\.\S+|'          # Email addresses
    r'\b[a-fA-F0-9]{10,}\b|'   # Long hex strings (>= 10 chars)
    r'(?:[a-zA-Z]:)?(?:[\\/][^\\/\n]+)+[\\/]?|' # Paths (improved slightly)
    r'\S+\.(?:com|nl|org|net|eu|be|io|dev|app|info|biz)\b/?\S*|' # Common TLDs
    r'\S*\.(?:txt|gz|js|css|php|html|xml|json|yaml|log|pdf|docx?|xlsx?|zip|rar|jpe?g|png|gif|svg|mp[34]|avi|mov|iso|img|exe|dll|so)\b|' # More file extensions
    r'\{[^{}]*\}|\[[^\[\]]*\]|<[^<>]*>|' # Things in brackets/tags
    r'(\d{1,3}\.){3}\d{1,3}|'   # IP addresses
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b|' # Month names (often in logs/metadata)
    r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|' # Date formats
    r'__\w+__|'               # Dunder methods/vars often in code examples
    r'\.{3,}'                 # Ellipsis with 3 or more dots
)
# Characters often found in noisy text (beyond basic punctuation)
EXCESSIVE_CHARS = set('|»«•·{}[]<>\\^~*') # Removed digits, handled by ratio check


# --- Helper Functions ---

def is_clean_sentence(text: str) -> bool:
    """
    Checks if a sentence is likely clean enough for TTS.
    Returns True if clean, False otherwise.
    """
    if not text: return False
    char_count = len(text)
    if char_count < MIN_CHARS: return False

    # 1. Check for bad patterns (Regex) - often faster to do first
    if BAD_PATTERNS.search(text): return False

    # 2. Check for bad keywords (case-insensitive)
    lower_text = text.lower()
    if any(keyword in lower_text for keyword in BAD_KEYWORDS): return False

    # 3. Check character ratios
    alpha_count = 0
    upper_count = 0
    digit_count = 0
    punct_count = 0
    excessive_char_found = False

    for char in text:
        if char.isalpha():
            alpha_count += 1
            if char.isupper():
                upper_count += 1
        elif char.isdigit():
            digit_count += 1
        elif char in string.punctuation:
            punct_count += 1
        # Check for specific excessive chars only if needed
        if not excessive_char_found and char in EXCESSIVE_CHARS:
             excessive_char_found = True

    if excessive_char_found: return False

    # Ratio checks (avoid division by zero)
    if char_count == 0: return False # Should be caught by MIN_CHARS but safe check
    if alpha_count / char_count < MIN_ALPHA_RATIO: return False # Too few letters
    if digit_count / char_count > MAX_DIGIT_RATIO: return False # Too many digits
    if alpha_count > 0 and upper_count / alpha_count > MAX_UPPER_RATIO: return False # Too many uppercase letters

    # 4. Check for repetitive characters
    counts = collections.Counter(text)
    if counts and counts.most_common(1)[0][1] / char_count > 0.5: return False # Highly repetitive

    # 5. Optional: Language Detection (if library installed and uncommented)
    # try:
    #     if detect(text) != 'nl':
    #         return False
    # except LangDetectException:
    #     return False # Treat detection failure as unclean

    return True # If none of the above checks failed


def get_phonemes(text: str) -> list:
    """
    Converts text to a list of phonemes using the espeak-ng command-line tool.
    """
    if not text: return []
    try:
        # Using --ipa=1 for broader IPA symbols, which might be better for some TTS
        command = ['espeak-ng', '-v', 'nl', '-q', '-x', '--ipa=1', text]
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        phonemes_str = result.stdout.strip()
        # Regex to capture IPA symbols, including those with diacritics or multi-character ones
        # This regex should capture all the target phonemes: j, d, f, a, ŋ, y, b, s
        # and other common Dutch IPA symbols.
        cleaned_phonemes = re.findall(r'[a-zA-Zəɛːɑɔøːœːɪŋɡʃʒɣχyːβθð]+', phonemes_str)
        return cleaned_phonemes
    except FileNotFoundError:
        print("Error: 'espeak-ng' command not found. Please ensure it's installed and in your PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # print(f"Warning: espeak-ng failed for text: '{text[:50]}...'. Error: {e}", file=sys.stderr)
        return [] # Fail silently on G2P errors for individual sentences
    except Exception as e:
        # print(f"Warning: An unexpected error occurred in get_phonemes for text: '{text[:50]}...'. Error: {e}", file=sys.stderr)
        return [] # Fail silently on other errors


def calculate_phoneme_density(phoneme_list, target_phonemes):
    """Calculates the density of target phonemes in a list."""
    if not phoneme_list: return 0.0
    target_count = sum(1 for p in phoneme_list if p in target_phonemes)
    return target_count / len(phoneme_list) if phoneme_list else 0.0


# --- Load OSCAR Dataset ---
try:
    print(f"Loading OSCAR dataset ({OSCAR_DATASET_NAME}, lang={OSCAR_LANGUAGE})...")
    oscar_dataset = load_dataset(
        OSCAR_DATASET_NAME, language=OSCAR_LANGUAGE, streaming=True,
        split="train", trust_remote_code=True
    )
    print("OSCAR dataset loaded successfully in streaming mode.")
except Exception as e:
    print(f"Error loading OSCAR dataset: {e}")
    sys.exit(1)


# --- Main Processing Logic ---
buffer_nl = []
buffer_fr = []
processed_count = 0
nl_total_found = 0
fr_total_found = 0
skipped_unclean = 0 # Counter for skipped sentences

print(f"\nStarting corpus processing from OSCAR stream...")
print(f"Targeting NL phonemes: {TARGET_PHONEMES_NL}")
print(f"Targeting FR phonemes: {TARGET_PHONEMES_FR}")
print(f"Density threshold NL: {DENSITY_THRESHOLD_NL}")
print(f"Density threshold FR: {DENSITY_THRESHOLD_FR}")
print(f"Sentence length limits: {MIN_WORDS}-{MAX_WORDS} words")
print(f"Cleaning limits: MIN_CHARS={MIN_CHARS}, MAX_UPPER={MAX_UPPER_RATIO*100}%, MIN_ALPHA={MIN_ALPHA_RATIO*100}%, MAX_DIGIT={MAX_DIGIT_RATIO*100}%")
if PROCESS_LIMIT: print(f"Processing limit set to {PROCESS_LIMIT} sentences.")
print(f"Save interval: {SAVE_INTERVAL} sentences")

# --- Ensure output files are empty before starting ---
try:
    with open(OUTPUT_SENTENCES_NL_RICH, 'w', encoding='utf-8') as f_nl, \
         open(OUTPUT_SENTENCES_FR_RICH, 'w', encoding='utf-8') as f_fr:
        pass # Just to create/truncate the files
    print("Output files cleared.")
except IOError as e:
    print(f"Error clearing output files: {e}")
    sys.exit(1)
# ---

try:
    progress_bar = tqdm(oscar_dataset, total=PROCESS_LIMIT, desc="Processing OSCAR", unit=" sentences")
    for item in progress_bar:
        sentence = item.get('text', '').strip()
        if not sentence: continue

        processed_count += 1 # Increment processed count early

        # --- Apply Cleaning Filter ---
        if not is_clean_sentence(sentence):
            skipped_unclean += 1
            continue # Skip to the next sentence if it's not clean
        # --- End Cleaning Filter ---

        word_count = len(sentence.split())
        if not (MIN_WORDS <= word_count <= MAX_WORDS): continue

        phonemes = get_phonemes(sentence)
        if not phonemes: continue

        # Check NL density
        density_nl = calculate_phoneme_density(phonemes, TARGET_PHONEMES_NL)
        if density_nl >= DENSITY_THRESHOLD_NL:
            buffer_nl.append(sentence)

        # Check FR density
        density_fr = calculate_phoneme_density(phonemes, TARGET_PHONEMES_FR)
        if density_fr >= DENSITY_THRESHOLD_FR:
            buffer_fr.append(sentence)

        # --- Periodic Saving ---
        if processed_count % SAVE_INTERVAL == 0:
            try:
                with open(OUTPUT_SENTENCES_NL_RICH, 'a', encoding='utf-8') as outfile_nl:
                    for s in buffer_nl: outfile_nl.write(s + '\n')
                with open(OUTPUT_SENTENCES_FR_RICH, 'a', encoding='utf-8') as outfile_fr:
                    for s in buffer_fr: outfile_fr.write(s + '\n')
                nl_total_found += len(buffer_nl)
                fr_total_found += len(buffer_fr)
                buffer_nl = []
                buffer_fr = []
                progress_bar.set_postfix(nl=nl_total_found, fr=fr_total_found, skip=skipped_unclean, refresh=True)
            except IOError as e:
                print(f"\nError saving checkpoint at {processed_count} sentences: {e}")
        else:
             # Update progress bar description less frequently to reduce overhead
             if processed_count % 500 == 0: # Example: update every 500
                 progress_bar.set_postfix(nl=nl_total_found + len(buffer_nl), fr=fr_total_found + len(buffer_fr), skip=skipped_unclean, refresh=False)


        # Optional: Stop after processing a certain number of sentences
        if PROCESS_LIMIT and processed_count >= PROCESS_LIMIT:
            print(f"\nReached processing limit of {PROCESS_LIMIT} sentences.")
            break
    progress_bar.close()

except Exception as e:
    print(f"\nAn error occurred during processing: {e}")
    import traceback
    traceback.print_exc()

# --- Final Save ---
try:
    print(f"\nPerforming final save...")
    with open(OUTPUT_SENTENCES_NL_RICH, 'a', encoding='utf-8') as outfile_nl:
        for s in buffer_nl: outfile_nl.write(s + '\n')
    with open(OUTPUT_SENTENCES_FR_RICH, 'a', encoding='utf-8') as outfile_fr:
        for s in buffer_fr: outfile_fr.write(s + '\n')
    nl_total_found += len(buffer_nl)
    fr_total_found += len(buffer_fr)
    print("Final save complete.")
except IOError as e:
    print(f"Error during final save: {e}")

print(f"\nProcessing finished. Processed {processed_count} sentences.")
print(f"Skipped {skipped_unclean} unclean sentences.") # Report skipped count
print(f"Total found NL-rich sentences: {nl_total_found}")
print(f"Total found FR-rich sentences: {fr_total_found}")
print(f"Results saved to {OUTPUT_SENTENCES_NL_RICH} and {OUTPUT_SENTENCES_FR_RICH}")

print("\nScript complete.")
