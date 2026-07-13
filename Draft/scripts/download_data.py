import os
from datasets import load_dataset
import pandas as pd

def download_mt560():
    print("Downloading MT560 English-Dholuo dataset...")
    dataset = load_dataset("michsethowusu/english-dholuo_sentence-pairs_mt560")
    
    # The dataset usually has a 'train' split
    df = pd.DataFrame(dataset['train'])
    
    # Save to CSV
    output_path = "data/mt560_raw.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    
    # Basic cleaning: deduplicate and remove empty rows
    df_clean = df.drop_duplicates().dropna()
    clean_path = "data/mt560_clean.csv"
    df_clean.to_csv(clean_path, index=False)
    print(f"Saved {len(df_clean)} cleaned rows to {clean_path}")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    download_mt560()
