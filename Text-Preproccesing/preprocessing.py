import re
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from pyvi import ViTokenizer

# Initialize tools
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
spell_checker = SpellChecker()
stop_words = set(stopwords.words('english'))

vietnamese_stopwords = {
    "và", "là", "có", "của", "cho", "với", "được", "trong", "này", "một", "đó", "không",
    "cũng", "nhưng", "nếu", "thì", "để", "các", "điều"
}
stop_words.update(vietnamese_stopwords)

def correct_spelling(text):
    words = text.split()
    corrected_words = [spell_checker.correction(word) if word in spell_checker else word for word in words]
    return ' '.join(corrected_words)

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


#remocve frequent and rare words
def remove_frequent_rare_words(text, frequent_words, rare_words):
    words = text.split()
    words = [word for word in words if word not in frequent_words and word not in rare_words]
    return ' '.join(words)
