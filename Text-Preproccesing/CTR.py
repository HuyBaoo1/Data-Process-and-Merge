import pandas as pd
import numpy as np
import re
from collections import defaultdict
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import nltk
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from pyvi import ViTokenizer
from spellchecker import SpellChecker

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Define file path
file_path = "C:/Users/ASUS/Downloads/Gửi Huy.xlsx"

# Read the Excel file
df = pd.read_excel(file_path, sheet_name=1)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
spell_checker = SpellChecker()
stop_words = set(stopwords.words('english'))

vietnamese_stopwords = {
    "và", "là", "có", "của", "cho", "với", "được", "trong", "này", "một", "đó", "không",
    "cũng", "nhưng", "nếu", "thì", "để", "này", "các", "điều"
}

stop_words.update(vietnamese_stopwords)

# Function to correct spelling
def correct_spelling(text):
    words = text.split()
    corrected_words = [spell_checker.correction(word) if word in spell_checker else word for word in words]
    return ' '.join(corrected_words)

# Function to normalize keywords
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\sàáạãảâấầậẫẩăắằặẵẳèéẹẽẻêếềệễểìíịĩỉòóọõỏôốồộỗổơớờợỡởùúụũủưứừựữửỳýỵỹỷđ]', '', text)
    text = correct_spelling(text)
    text = ViTokenizer.tokenize(text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Normalize keywords
df['normalized_keywords'] = df['keyword'].astype(str).apply(normalize_text)

# Remove keywords with more than 8 digits
df = df[~df['keyword'].astype(str).str.match(r'^\d{9,}$', na=False)]

# Remove spamming keywords
df = df[~df['keyword'].astype(str).str.match(r'(.)\1{3,}', na=False)]

# Compute word frequencies
word_freq = pd.Series(' '.join(df['normalized_keywords']).split()).value_counts()
frequent_words = set(word_freq[word_freq > 0.95 * len(df)].index)
rare_words = set(word_freq[word_freq < 5].index)

def remove_frequent_rare_words(text):
    words = text.split()
    words = [word for word in words if word not in frequent_words and word not in rare_words]
    return ' '.join(words)

df['normalized_keywords'] = df['normalized_keywords'].apply(remove_frequent_rare_words)

# Group similar keywords
keyword_groups = defaultdict(list)
for keyword in df['normalized_keywords'].unique():
    matched = False
    for group in keyword_groups:
        if fuzz.ratio(keyword, group) > 80:
            keyword_groups[group].append(keyword)
            matched = True
            break
    if not matched:
        keyword_groups[keyword].append(keyword)

# Create a mapping
keyword_map = {variant: main for main, variants in keyword_groups.items() for variant in variants}
df["merged_keywords"] = df["normalized_keywords"].map(keyword_map)

# Aggregate data
df_grouped = df.groupby("merged_keywords").agg({
    "Searched Count": "sum",
    "Search-to-watch": "sum",
}).reset_index()

df_grouped['CTR'] = (df_grouped['Search-to-watch'] / df_grouped['Searched Count'].replace(0, np.nan)) * 100
df_grouped['CTR'] = df_grouped['CTR'].fillna(0)
df_grouped['CTR'] = df_grouped['CTR'].round(2)

# Set filtering thresholds
MIN_SEARCH_VOLUME = 10
MIN_CTR = 10

# Apply filters
filtered_df = df_grouped[(df_grouped['Searched Count'] >= MIN_SEARCH_VOLUME) & (df_grouped['CTR'] >= MIN_CTR)]
filtered_df.to_csv("filtered_data_1.csv", index=False)

# Save filtered data to the existing Excel file
existing_file_path = 'C:/Users/ASUS/Downloads/Gửi Huy.xlsx'
book = openpyxl.load_workbook(existing_file_path)
new_sheet_name = 'NewFilteredData(Test)'
sheet = book.create_sheet(title=new_sheet_name)
for r in dataframe_to_rows(filtered_df, index=False, header=True):
    sheet.append(r)
book.save(existing_file_path)
/project_root
 ├── data_processing/
 │   ├── load_data.py
 │   ├── save_data.py
 ├── preprocessing/
 │   ├── normalize_text.py
 │   ├── remove_frequent_rare_words.py
 ├── feature_extraction/
 │   ├── frequency_vector.py
 │   ├── tfidf_vectorizer.py
 │   ├── word_embeddings.py
 ├── keyword_grouping/
 │   ├── group_keywords.py
 ├── main.py
