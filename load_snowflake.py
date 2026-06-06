import pandas as pd
import numpy as np
import snowflake.connector
import os
from dotenv import load_dotenv

load_dotenv()

data_dir = r"C:\Users\IZEROUAL\Desktop\MSPR3\data"

conn = snowflake.connector.connect(
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
)
cursor = conn.cursor()
print("Connecte a Snowflake !")


def clean_val(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if str(v).strip() in ("ND", "nan", "NaN", "NAN", ""):
        return None
    return v


total = 0
for filename in sorted(os.listdir(data_dir)):
    if filename.endswith(".csv") and "eco2mix" in filename.lower():
        csv_path = os.path.join(data_dir, filename)
        print(f"\nChargement : {filename}...")

        with open(csv_path, encoding="utf-8-sig") as f:
            first_line = f.readline()
        sep = "\t" if first_line.count("\t") > first_line.count(";") else ";"

        df = pd.read_csv(csv_path, sep=sep, encoding="utf-8-sig", dtype=str)
        df = df.iloc[:, :40]

        print(f"  -> {len(df)} lignes, {len(df.columns)} colonnes")

        rows = [
            tuple(clean_val(v) for v in row)
            for row in df.itertuples(index=False, name=None)
        ]

        placeholders = ",".join(["%s"] * 40)
        batch_size = 1000
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            cursor.executemany(
                f"INSERT INTO ECO2MIX_DATA VALUES ({placeholders})", batch
            )
            conn.commit()
            print(f"  -> {min(i + batch_size, len(rows))}/{len(rows)} lignes...")

        print(f"  OK : {len(rows)} lignes")
        total += len(rows)

print(f"\nTotal : {total} lignes inserees !")
cursor.close()
conn.close()
