import pandas as pd
import sqlite3
import os
from glob import glob

# Connect to SQLite
conn = sqlite3.connect("bgg.db")

# Loop through CSV files
for csv_file in glob(os.path.join("data","*.csv")):
    if "documentation" in csv_file:
        continue  # skip docs
    table_name = csv_file.replace(".csv", "")
    df = pd.read_csv(csv_file)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Imported {table_name} ({len(df)} rows)")

conn.close()
print("All CSVs successfully imported into bgg.db")