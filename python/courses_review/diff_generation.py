import pandas as pd
import numpy as np
import csv
import difflib
from pprint import pprint

#We can use the "robust algorithm" from gemini to read up and clean data, then export it and make the diff.


df = pd.read_csv('user_courses_review_09_2023.csv', on_bad_lines="skip")

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

# Drop rows where the rating is truly missing (unusable for sentiment)
df_clean = df.dropna(subset=['review_rating']).copy()








df_clean.to_csv('read_lines.csv', quoting=csv.QUOTE_MINIMAL, quotechar="\"", index=False, doublequote=True)
print(df_clean.shape)
print(df_clean.describe())
print(df_clean.info())



with open ('user_courses_review_09_2023.csv', "r", encoding="utf-8") as original:
    original_lines = original.readlines()
with open ('read_lines.csv', "r", encoding="utf-8") as readable:
    readable_lines = readable.readlines()

d = difflib.Differ()
result = list(d.compare(original_lines, readable_lines))
pprint(result[1:20])

print("It didn't work: not having Pandas writing the same quoting style results in differences in almost every line.")