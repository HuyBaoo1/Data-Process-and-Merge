from .load_data import load_data
from .save_data import save_filtered_data

def process_data(file_path):
    """Output cleaned data for feature extraction"""
    df = load_data(file_path)
    
    # Existing preprocessing
    df['normalized'] = df['keyword'].astype(str).apply(normalize_text)
    df = df[~df['keyword'].str.match(r'^\d{9,}$|(.)\1{3,}', na=False)]
    
    # Return standardized format
    return df[['keyword', 'normalized', 'Searched Count']]