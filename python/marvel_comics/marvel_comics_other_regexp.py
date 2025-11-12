import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import zipfile
from io import BytesIO

# The file ID for the uploaded ZIP file
zip_file_id = "Marvel_Comics.zip"

# Read the CSV directly from the ZIP content
df = pd.read_csv(zip_file_id,compression='zip')

# Display initial information
print("--- Initial DataFrame Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\n--- Column Information and Missing Values ---")
df.info()
print("\n--- First 5 Rows ---")
print(df.head())


# Clean Price
# Assuming 'df' is your DataFrame

# 1. Handle missing values (NaN) by filling them with a string that will become 0.00
df['Price'] = df['Price'].fillna('$0.00')

# 2. Clean the string: remove the dollar sign and surrounding whitespace
df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False).str.strip()

# 3. Replace the string 'Free' with the numeric string '0.00'
df['Price'] = df['Price'].replace('Free', '0.00')

# 4. Convert the cleaned string column to a numeric (float) type.
# Any value that couldn't be converted after the previous steps would be set to NaN (if errors='coerce').
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# 5. (Optional but Recommended) Fill any remaining NaNs (if introduced by errors='coerce') with 0.00
df['Price'] = df['Price'].fillna(0.00)

# The df DataFrame is already loaded and the column 'publish_date' was processed previously.

# Standardize the format (already done in the preprocessing by converting to datetime)
# The standard format is the internal pandas/numpy datetime type.
# We will check for NaT values which indicate non-compliant (unparsable) original strings.

# Create a boolean mask for rows where the date could not be parsed (is NaT)
non_compliant_mask = df['publish_date'].isna()

# Extract the original date strings that failed to convert
non_compliant_dates = df[non_compliant_mask]['publish_date'].index

# Report the non-compliant dates (by index)
print("--- Non-Compliant Date Report ---")
if non_compliant_dates.empty:
    print("✅ All dates were successfully standardized into the datetime format.")
else:
    print(f"❌ Found {len(non_compliant_dates)} non-compliant/unparsable date entries.")
    # Show the original index of the rows that failed
    print("\nIndices of rows with non-compliant dates:")
    print(non_compliant_dates)
    # Display the rows for inspection (showing the original index for reference)
    print("\nRows with Unparsable Dates (Original Index, Issue Title, Original Date String):")
    print(df[non_compliant_mask][['issue_title', 'publish_date']].head(10))


# 1. CRITICAL STEP: Convert the column to datetime format
# The 'errors='coerce' will turn non-compliant strings into NaT (Not a Time)
df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')

# 2. Check for NaT values (non-compliant/unparsable dates)
non_compliant_mask = df['publish_date'].isna()

# 3. Standardize to a common display format (e.g., YYYY-MM-DD)
# The .dt accessor will now work correctly
df['publish_date_standard'] = df['publish_date'].dt.strftime('%Y-%m-%d')

print("\n--- Standardization Check ---")
print(f"New column 'publish_date_standard' created with format YYYY-MM-DD.")
print(df[['publish_date', 'publish_date_standard']].head())

# You can still use the report logic to see if any dates failed the conversion:
print(f"Non-compliant/unparsable entries found: {non_compliant_mask.sum()}")

# plot active comics per year

# The DataFrame 'df' is assumed to be loaded and ready from previous steps.

# 1. Function to extract all years from the active_years string
def extract_years(years_string):
    if pd.isna(years_string):
        return []
    # Use regex to find all four-digit numbers enclosed in parentheses
    # Example: (2006),(2006) -> ['2006', '2006']
    # Example: (2016) -> ['2016']
    years = re.findall(r'\((\d{4})\)', str(years_string))
    # Convert to a set to count unique active years per comic, then back to a list
    return list(set(years))

# Apply the function to the column
all_years_list = df['active_years'].apply(extract_years).explode()

# 2. Count Occurrences (Count the number of active comic series per year)
# Drop any NaN results from the initial apply/explode
active_years_counts = all_years_list.dropna().value_counts().sort_index()

# 3. Create Time-Series Data
active_years_df = active_years_counts.reset_index()
active_years_df.columns = ['Year', 'Active Comic Count']

# Convert 'Year' to integer for proper sorting and plotting
active_years_df['Year'] = active_years_df['Year'].astype(int)

print("--- Data Ready for Plotting ---")
print(active_years_df.head())
print("\n--- Summary of Active Years ---")
print(active_years_df.describe())

# 4. Generate the Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=active_years_df, x='Year', y='Active Comic Count', marker='o')

plt.title('Marvel Comic Series Activity Over Time (By Publication Year)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Active Comic Series', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# The 'df' DataFrame is assumed to be loaded with 'publish_date' as datetime
# and 'Price' as a clean float (where Free/missing are 0.00).

# 1. Extract the year from the datetime object
df['Year'] = df['publish_date'].dt.year

# 2. Filter out issues where the price is 0.00 (as these are 'Free' or missing, which would skew the average price)
# If we include $0.00, the average price would be artificially lowered.
priced_issues_df = df[df['Price'] > 0]

# 3. Calculate the average price for each year
average_price_per_year = priced_issues_df.groupby('Year')['Price'].mean().reset_index()
average_price_per_year.columns = ['Year', 'Average Price']

# 4. Filter for years with a sufficient number of publications (e.g., at least 5) to ensure stability, if necessary.
# Let's keep all years for now unless the plot looks too erratic.

print("--- Data Ready for Price Evolution Plotting ---")
print(average_price_per_year.tail())

# 5. Generate the Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=average_price_per_year, x='Year', y='Average Price', marker='o')

plt.title('Evolution of Average Marvel Comic Issue Price Over Time', fontsize=16)
plt.xlabel('Publication Year', fontsize=12)
plt.ylabel('Average Price (USD)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# --- 1. Data Loading and Initial Cleaning ---

# NOTE: The CSV file needs to be read from the uploaded ZIP file path
file_name = "Marvel_Comics.zip"
df = pd.read_csv(file_name,compression='zip')

# Date conversion (Critical for time analysis)
df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
df['Year'] = df['publish_date'].dt.year.astype('Int64')

# Combine relevant text columns (title and description) and convert to lowercase
df['full_text'] = (df['issue_title'].fillna('') + " " + df['issue_description'].fillna('')).str.lower()

# --- 2. Define Combined Topics to Track ---
# These topics are chosen based on the results from the previous unigram/bigram
# analysis to represent the core intellectual property (IP).
topics_to_track = [
    'spider-man',
    'x-men',
    'avengers',
    'captain america',
    'iron man',
    'hulk',
    'wolverine',
    'deadpool',
    'war',
    'world'
]

# --- 3. Track Frequencies (Create Binary Columns) ---
for topic in topics_to_track:
    # Use str.contains() with word boundaries (\b) and re.escape to accurately find the topic phrase.
    # This prevents partial matches and ensures we find the whole concept.
    # The 're.escape' handles hyphens correctly (e.g., in 'spider-man').
    df[f'kw_{topic}'] = df['full_text'].str.contains(r'\b' + re.escape(topic) + r'\b', case=False, na=False)

# --- 4. Group and Sum Occurrences ---
topic_cols = [f'kw_{t}' for t in topics_to_track]

# Group by Year and sum the occurrences of the topics
topic_evolution_df = df.dropna(subset=['Year']).groupby('Year')[topic_cols].sum()

# Convert counts to float and reset index for easier plotting
topic_evolution_df = topic_evolution_df.astype(float).reset_index()

# Clean up column names for the legend
topic_evolution_df.columns = ['Year'] + [col.replace('kw_', '').title() for col in topic_evolution_df.columns[1:]]

# --- 5. Filter Data for Plotting ---
# Filter out the extremely low count years at the end (e.g., 2018 onwards) which are often incomplete data
topic_evolution_df_filtered = topic_evolution_df[topic_evolution_df['Year'] < 2018].set_index('Year')

# --- 6. Generate the Plot ---
plt.figure(figsize=(14, 8))

# Plot each column (each topic) as a separate line
topic_evolution_df_filtered.plot(ax=plt.gca(), linewidth=2)

plt.title('Evolution of Top 10 Combined Marvel Topics Over Time', fontsize=16)
plt.xlabel('Publication Year', fontsize=12)
plt.ylabel('Annual Count of Topic Mention in Descriptions', fontsize=12)
plt.legend(title='Core Topics', loc='upper left', bbox_to_anchor=(1.01, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading and Initial Cleaning ---
# NOTE: Replace 'Marvel_Comics.zip/Marvel_Comics.csv' with the correct path
# if you are running this outside of a virtual environment.
file_name = "Marvel_Comics.zip/Marvel_Comics.csv"
df = pd.read_csv(file_name)

# Date conversion (Critical for time analysis)
df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
df['Year'] = df['publish_date'].dt.year.astype('Int64')

# Combine relevant text columns (title and description)
# IMPORTANT: Do NOT convert to lowercase yet, as we need the original capitalization
df['full_text_original'] = df['issue_title'].fillna('') + " " + df['issue_description'].fillna('')

# --- 2. Define Stop Words and Filter Capitalized Words ---
# Define common capitalized stop words that appear frequently but offer no insight
stop_words_cap = set([
    'THE', 'A', 'AN', 'IS', 'IT', 'TO', 'OF', 'IN', 'AND', 'FOR', 'WITH', 'ON', 'THIS',
    'S', 'WILL', 'NEW', 'ARE', 'WHO', 'FROM', 'BUT', 'NOT', 'AS', 'AT', 'I', 'HE', 'SHE',
    'THEY', 'WE', 'BE', 'BY', 'UP', 'DOWN', 'OUT', 'INTO', 'ISSUE', 'COMIC', 'VS',
    'PGS', 'STORY', 'CAN', 'ONE', 'GET', 'MUST', 'HAS', 'HAVE', 'ALL', 'MORE', 'SERIES',
    'LIMITED', 'BACK', 'FIRST', 'TWO', 'WHAT', 'DO', 'GO', 'WHEN', 'FIND', 'RE', 'T',
    'MAKE', 'JUST', 'TIME', 'TAKE', 'WAY', 'HIM', 'HER', 'THEIR', 'ITS', 'NONE', 'YOU',
    'THAT', 'MARVEL', 'MAX', 'AGE', 'VOL', 'SUGGESTED', 'PSR' # Added general terms
])

# Function to extract only ALL-CAPS words (length > 2)
def extract_all_caps_words(text):
    # Find all words consisting only of uppercase letters and non-word characters
    all_caps = re.findall(r'\b[A-Z\'-]+\b', str(text))
    # Filter out stop words and ensure it's strictly ALL-CAPS
    return [word for word in all_caps if word == word.upper() and word not in stop_words_cap and len(word) > 2]

# Apply the function to the text column
all_caps_words = df['full_text_original'].apply(extract_all_caps_words).explode().dropna()

# --- 3. Identify Top Capitalized Concepts ---
# Get the top 10 most emphasized concepts
capitalized_word_counts = all_caps_words.value_counts()
top_keywords = capitalized_word_counts.head(10).index.tolist()

# --- 4. Prepare data for Time Analysis ---
# Create binary columns for tracking the top keywords
for keyword in top_keywords:
    # Use str.contains() to check for the EXACT ALL-CAPS word (case=True)
    df[f'kw_{keyword}'] = df['full_text_original'].str.contains(r'\b' + re.escape(keyword) + r'\b', case=True, na=False)

# Group by Year and sum the occurrences of the top keywords
topic_cols = [f'kw_{k}' for k in top_keywords]
topic_evolution_df = df.dropna(subset=['Year']).groupby('Year')[topic_cols].sum()

# Clean up column names for the legend
topic_evolution_df.columns = [col.replace('kw_', '').title() for col in topic_evolution_df.columns]

# --- 5. Filter Data for Plotting ---
# Filter out the extremely low count years at the end (2018 onwards)
topic_evolution_df_filtered = topic_evolution_df[topic_evolution_df.index < 2018]

# --- 6. Generate the Plot ---
plt.figure(figsize=(14, 8))

# Plot each column (each capitalized concept) as a separate line
topic_evolution_df_filtered.plot(ax=plt.gca(), linewidth=2)

plt.title('Evolution of Top 10 ALL-CAPS Marvel Concepts Over Time', fontsize=16)
plt.xlabel('Publication Year', fontsize=12)
plt.ylabel('Annual Count of ALL-CAPS Topic Mention in Descriptions', fontsize=12)
plt.legend(title='Emphasized Concepts', loc='upper left', bbox_to_anchor=(1.01, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

