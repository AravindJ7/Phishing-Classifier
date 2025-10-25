import pandas as pd
from sklearn.utils import shuffle

# --- Configuration ---
SAFE_FILE = 'safe_emails.csv'
PHISHING_FILE = 'Phishing_Email.csv' # The file you just downloaded
FINAL_OUTPUT_FILE = 'final_model_dataset.csv'
# ---------------------

print("Starting dataset combination process...")

try:
    # --- 1. Load your SAFE (ham) emails ---
    print(f"Loading safe emails from {SAFE_FILE}...")
    # Your safe_emails.csv has 'Subject' and 'Message'
    safe_df = pd.read_csv(SAFE_FILE, low_memory=False)
    
    # Combine Subject and Message into a single 'text' column
    # We fillna('') to handle any missing subjects or messages
    safe_df['text'] = safe_df['Subject'].fillna('') + ' ' + safe_df['Message'].fillna('')
    
    # Keep only the text and add the final label (0 = safe)
    safe_df = safe_df[['text']]
    safe_df['label'] = 0
    print(f"Loaded {len(safe_df)} safe emails.")


    # --- 2. Load the PHISHING emails ---
    print(f"Loading phishing emails from {PHISHING_FILE}...")
    phish_df = pd.read_csv(PHISHING_FILE, low_memory=False)
    
    # Filter to get ONLY the phishing emails
    # The column is 'Email Type' and the label is 'Phishing Email'
    phish_df = phish_df[phish_df['Email Type'] == 'Phishing Email'].copy()
    
    # Rename 'Email Text' to 'text' to match our safe_df
    phish_df.rename(columns={'Email Text': 'text'}, inplace=True)
    
    # Keep only the text and add the final label (1 = phishing)
    phish_df = phish_df[['text']]
    phish_df['label'] = 1
    print(f"Loaded {len(phish_df)} phishing emails.")


    # --- 3. Combine and Finalize ---
    print("Combining datasets...")
    # Concatenate the two dataframes into one
    final_df = pd.concat([safe_df, phish_df], ignore_index=True)
    
    # Shuffle the dataset randomly
    final_df = shuffle(final_df, random_state=42).reset_index(drop=True)
    
    # Drop any rows where the 'text' might be empty
    final_df.dropna(subset=['text'], inplace=True)

    # Save the final, ready-to-use dataset
    final_df.to_csv(FINAL_OUTPUT_FILE, index=False)
    
    print("\nSuccess! 🎉")
    print(f"Total emails in final dataset: {len(final_df)}")
    print(f"Final dataset saved to: {FINAL_OUTPUT_FILE}")
    print("\n--- Final Label Distribution ---")
    print(final_df['label'].value_counts())


except FileNotFoundError as e:
    print(f"\nError: File not found.")
    print(f"Please make sure '{e.filename}' is in your ML_Project directory.")
except KeyError as e:
    print(f"\nError: A column name was not found: {e}")
    print("The CSV file structure might have changed. Please double-check column names.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
