#remocve frequent and rare words
def remove_frequent_rare_words(text, frequent_words, rare_words):
    words = text.split()
    words = [word for word in words if word not in frequent_words and word not in rare_words]
    return ' '.join(words)