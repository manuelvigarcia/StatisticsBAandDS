import pandas as pd
import re
from collections import Counter

# Load the data
data = pd.read_csv('Marvel_Comics.csv')

# Display the first few rows of the dataset
print(data.head(15))

def extract_capitalized_words(text):
    # Use regex to find sequences of capitalized words
    return re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', str(text))

# Combine the comic_name and issue_title columns into a single text
combined_text = data['comic_name'].str.cat(data['issue_title'], sep=' ').str.cat(data['issue_description'], sep=' ')

# Extract capitalized words from the combined text
capitalized_words = combined_text.apply(extract_capitalized_words).explode().dropna()

# Count the occurrences of each capitalized word/sequence
word_counts = Counter(capitalized_words)

# Get the top 10 most common capitalized words/sequences
most_common_words = word_counts.most_common(10)
print(most_common_words)
