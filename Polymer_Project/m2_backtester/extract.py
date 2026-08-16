import pandas as pd
import os
from glob import glob


def extract_data(file_path):
    """
    Extract data from a CSV file and return it as a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The extracted data as a DataFrame.
    """
    try:
        data = pd.read_pickle(file_path)
        data.columns = [col[0] for col in data.columns.to_flat_index()]
        data['indice'] = os.path.basename(file_path).split('_')[1]
        return data
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the Pickle file: {e}")
    

def main():
    all_files = glob(os.path.join("OneDrive_147_15-01-2026", "Yahoo_*", "*.pkl"))
    
    all_data = []
    for file_path in all_files:
        data = extract_data(file_path)
        all_data.append(data)

    df_final = pd.concat(all_data, ignore_index=True)
    print(df_final.head())
    df_final.to_csv("extracted_data.csv", index=False)


if __name__ == "__main__":
    main()
