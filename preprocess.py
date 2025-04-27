import pandas as pd

# Load the scraped CSV
df = pd.read_csv("formatted_tax_cases.csv")

# Define the 8 structured section fields
section_fields = [
    "Facts", "Issues", "PetArg", "RespArg",
    "Section", "Precedent", "CDiscource", "Conclusion"
]

# Drop rows where all section fields are empty or null
df_filtered = df.dropna(subset=section_fields, how='all')

# Fill remaining NaNs in all columns with empty strings
df_filtered.fillna('', inplace=True)

# Optional: Reset index
df_filtered.reset_index(drop=True, inplace=True)

# Save the cleaned CSV
df_filtered.to_csv("filtered_tax_cases.csv", index=False)

print(f"✅ Filtered dataset saved to 'filtered_tax_cases.csv'")
print(f"🧹 Original rows: {len(df)}, Filtered rows: {len(df_filtered)}")