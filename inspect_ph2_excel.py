from pathlib import Path
import pandas as pd

PH2_DIR = Path(r"C:\Users\HP\Desktop\Melonma\data\raw\PH2\PH2Dataset")
excel_file = PH2_DIR / "PH2_dataset.xlsx"

df = pd.read_excel(excel_file, header=None)

print("=" * 70)
print("PH2 COMPLETE EXCEL INSPECTION")
print("=" * 70)

print("Shape:", df.shape)

# Print every row that contains useful text
for i, row in df.iterrows():

    values = []

    for value in row:
        if pd.notna(value):
            values.append(str(value))

    if values:
        print(f"\nROW {i}:")
        print(" | ".join(values))