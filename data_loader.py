import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Convert necessary columns to string for vector DB
    df['student_id'] = df['student_id'].astype(str)
    return df