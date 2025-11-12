import pandas as pd

# Load dataset
file_path = 'patients.csv'
df = pd.read_csv(file_path)

# Step 1: Fill missing diagnosis with "Unknown"
df['Diagnosis'] = df['Diagnosis'].fillna('Unknown')

# Step 2: Standardize Diagnosis entries (title case, strip spaces)
df['Diagnosis'] = df['Diagnosis'].str.strip().str.title()

# (Optional) Fix common typos if any - can extend dictionary as needed:
def fix_typo(diagnosis):
    corrections = {
        'Common cold': 'Common Cold',
        'Diabetes ': 'Diabetes',
        'Flu ': 'Flu'
    }
    return corrections.get(diagnosis, diagnosis)

df['Diagnosis'] = df['Diagnosis'].apply(fix_typo)

# Step 3: Validate numeric ranges (example ranges)
age_min, age_max = 0, 120
lab_min, lab_max = 0, 300  # Adjust based on domain knowledge

out_of_range_age = df[(df['Age'] < age_min) | (df['Age'] > age_max)]
out_of_range_lab = df[(df['LabResult'] < lab_min) | (df['LabResult'] > lab_max)]

# Print any out-of-range rows (optional)
print("Out-of-range Age rows:\n", out_of_range_age)
print("Out-of-range LabResult rows:\n", out_of_range_lab)

# Step 4: Check for duplicate rows
duplicates = df.duplicated(keep=False)
duplicate_rows = df[duplicates]
print("Duplicate rows:\n", duplicate_rows)

# Export the cleaned/preprocessed dataset
export_path = 'patients_preprocessed_final.csv'
df.to_csv(export_path, index=False)
print(f"Preprocessed dataset saved to {export_path}")

# Define custom label order
label_order = ['Unknown', 'Common Cold', 'Flu', 'Diabetes', 'Hypertension']
label_mapping = {label: idx for idx, label in enumerate(label_order)}

# Apply label encoding
df['Diagnosis_LabelEncoded'] = df['Diagnosis'].map(label_mapping)

# Sample output
print(df[['Diagnosis', 'Diagnosis_LabelEncoded']].head())

# Export the cleaned/preprocessed/encoded dataset
export_path = 'patients_with_label_encoded.csv'
df.to_csv(export_path, index=False)
print(f"Preprocessed and recode dataset saved to {export_path}")
