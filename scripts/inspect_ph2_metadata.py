import pandas as pd
from pathlib import Path

excel_path = Path("data/raw/PH2/PH2Dataset/PH2_dataset.xlsx")

print("=" * 80)
print("PH2 DATASET - RAW EXCEL STRUCTURE")
print("=" * 80)

df = pd.read_excel(
    excel_path,
    sheet_name=0,
    header=None
)

print(f"\nExcel shape: {df.shape}")

print("\nROWS 1-30:")
print("-" * 80)

for index, row in df.iloc[:30].iterrows():
    values = []

    for value in row:
        if pd.notna(value):
            values.append(str(value))

    if values:
        print(f"ROW {index + 1}:")
        print(" | ".join(values))

print("\n" + "=" * 80)
print("POSSIBLE IMD ROWS")
print("=" * 80)

for index, row in df.iterrows():

    row_text = " | ".join(
        str(value)
        for value in row
        if pd.notna(value)
    )

    if "IMD" in row_text.upper():
        print(f"ROW {index + 1}: {row_text}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)