import os
import sys

# --- Configuratie ---
# Pas dit pad aan naar de locatie van je JASMIN data
jasmin_base_dir = "jasmin-data" # Relatief pad, of gebruik een absoluut pad

# Gegevens van de problematische opname
speaker_id = "N000214"
file_root = "fn000558"
component = "comp-q" # Aanname, controleer dit eventueel
region = "nl" # Afgeleid van 'N' in speaker_id

# Construeer het pad naar het .ort bestand
ort_file_name = f"{file_root}.ort"
ort_file_path = os.path.join(
    jasmin_base_dir,
    "Data", "data", "annot", "text", "ort",
    component,
    region,
    ort_file_name
)
ort_file_path = os.path.normpath(ort_file_path) # Maakt het pad netjes

print(f"Proberen .ort bestand te lezen: {ort_file_path}")

# --- Lees en print het bestand ---
try:
    # Probeer verschillende encodings, UTF-8 en Latin-1 zijn gebruikelijk
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1']
    file_content = None
    detected_encoding = None

    for enc in encodings_to_try:
        try:
            with open(ort_file_path, 'r', encoding=enc) as f:
                file_content = f.read()
            detected_encoding = enc
            print(f"Bestand succesvol gelezen met encoding: {detected_encoding}")
            break # Stop na succesvol lezen
        except UnicodeDecodeError:
            continue # Probeer volgende encoding
        except Exception as e:
            # Rapporteer andere leesfouten, maar ga door met proberen
             print(f"Fout bij lezen met encoding {enc}: {e}", file=sys.stderr)


    if file_content is not None:
        print("\n--- Inhoud van het .ort bestand ---")
        print(file_content)
        print("--- Einde inhoud ---")
    else:
        print(f"\nKon het bestand niet lezen met de geprobeerde encodings.")

except FileNotFoundError:
    print(f"\nFout: Bestand niet gevonden op het opgegeven pad: {ort_file_path}")
    print("Controleer of het pad en de bestandsnaam correct zijn.")
except Exception as e:
    print(f"\nEr is een onverwachte fout opgetreden: {e}")

