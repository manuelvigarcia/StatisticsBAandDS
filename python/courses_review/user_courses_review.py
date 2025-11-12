import csv
import pandas as pd

df = pd.read_csv('user_courses_review_09_2023.csv', on_bad_lines="skip")
df.to_csv('read_lines.csv', quoting=csv.QUOTE_MINIMAL, quotechar="\"", index=False, doublequote=True)
print(df.shape)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import csv

# Set plotting style
sns.set_theme(style="whitegrid")

# --- 1. Load Data with Robust Error Handling ---
file_name = "user_courses_review_09_2023_clean.csv"

# Load data by skipping bad lines to maximize usable data
try:
    df = pd.read_csv(file_name, on_bad_lines="skip")
except TypeError:
    df = pd.read_csv(file_name, error_bad_lines=False)
print(df.shape)
# --- 2. Data Cleaning/Prep (Salvaging Shifted Data) ---

# Convert review_rating to a nullable integer type, coercing non-numeric to NaN
df['review_rating'] = pd.to_numeric(df['review_rating'], errors='coerce').astype('Int64')

# Identify and Salvage Shifted Text (rows that loaded but had comment text in the rating column)
malformed_mask = df['review_rating'].isna()
df.loc[malformed_mask, 'review_comment'] = df.loc[malformed_mask, 'review_comment'].fillna('').astype(str) + ' ' + \
                                           df.loc[malformed_mask, 'review_rating'].astype(str)
df.loc[malformed_mask, 'review_rating'] = np.nan

# Clean remaining text fields
for col in ['course_name', 'lecture_name', 'review_comment']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('nan', '', regex=False)
print(df.shape)

# Drop rows where the rating is truly missing (unusable for sentiment)
df_clean = df.dropna(subset=['review_rating']).copy()
print(df_clean.shape)

# --- 3. Core Analysis & Highlights ---
min_reviews = 5

# a. Overall Sentiment
overall_avg_rating = df_clean['review_rating'].mean()

# b. Most Attended Course (by review count)
course_counts = df_clean.groupby('course_name')['review_rating'].count().sort_values(ascending=False)

# c. Most Commented Lecture (by non-empty review count)
lecture_comments = df_clean[df_clean['review_comment'].str.len() > 0].groupby('lecture_name')[
    'review_comment'].count().sort_values(ascending=False)

# d. Sentiment Analysis by Course/Lecture
course_sentiment = df_clean.groupby('course_name')['review_rating'].agg(['count', 'mean']).sort_values(by='mean',
                                                                                                       ascending=False)
course_sentiment_filtered = course_sentiment[course_sentiment['count'] >= min_reviews]
lecture_sentiment = df_clean.groupby('lecture_name')['review_rating'].agg(['count', 'mean']).sort_values(by='mean',
                                                                                                         ascending=False)
lecture_sentiment_filtered = lecture_sentiment[lecture_sentiment['count'] >= min_reviews]

# --- 4. Visualization: Rating Distribution ---
plt.figure(figsize=(8, 5))
df_clean['review_rating'].value_counts(sort=False).plot(kind='bar', color=sns.color_palette("viridis", 5))
plt.title('Distribution of Student Review Ratings', fontsize=14)
plt.xlabel('Rating (Stars)', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('rating_distribution.png')
plt.close()

# --- 5. Text Analysis (Simple Keyword Extraction) ---

# Custom, comprehensive local stop words list
stop_words_custom = set(['the', 'a', 'an', 'is', 'it', 'to', 'of', 'in', 'and', 'for', 'with', 'on', 'this',
                         'that', 'i', 'you', 'was', 'are', 'not', 'but', 'so', 'as', 'at', 'or', 'by', 'be',
                         'do', 'if', 'up', 'out', 'down', 'about', 'from', 'all', 'can', 'will', 'have', 'had',
                         'has', 'more', 'most', 'my', 'me', 'your', 'we', 'our', 'what', 'when', 'where', 'why',
                         'how', 'which', 'who', 'whom', 'here', 'there', 'he', 'she', 'they', 'them', 'their',
                         'his', 'her', 'its', 'just', 'too', 'only', 'very', 'much', 'many', 'get', 'like',
                         'know', 'see', 'thank', 'thanks', 'really', 'little', 'bit', 'would', 'should',
                         'could', 'one', 'two', 'use', 'using', 'used', 'course', 'lecture', 'class', 'great',
                         'excellent', 'good', 'very', 'i', 'you', 'was', 'are', 'not', 'but', 'so', 'it',
                         'this', 'that', 'from', 'all', 'can', 'will', 'module', 'content', 'much', 'better',
                         'make', 'well', 'time', 'things', 'examples', 'example', 'learn', 'learning',
                         'understand', 'information', 'knowledge', 'professor', 'instructor', 'teacher',
                         'easy', 'clear', 'help', 'helpful', 'get', 'best', 'topic', 'topics', 'data', 'concept'])


def get_keywords(comments, n=15):
    text = " ".join(comments)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = text.split()

    filtered_words = [word for word in words if word not in stop_words_custom and len(word) > 2]
    return Counter(filtered_words).most_common(n)


# Extract 5-star comments and 1/2-star (negative) comments
positive_comments = df_clean[df_clean['review_rating'] == 5]['review_comment'].tolist()
negative_comments = df_clean[df_clean['review_rating'].isin([1, 2])]['review_comment'].tolist()

top_positive_keywords = get_keywords(positive_comments, n=15)
top_negative_keywords = get_keywords(negative_comments, n=15)

# Save filtered sentiment data to CSVs for user inspection
course_sentiment_filtered.to_csv("course_sentiment_analysis.csv")
lecture_sentiment_filtered.to_csv("lecture_sentiment_analysis.csv")