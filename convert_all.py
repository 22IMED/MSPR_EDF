import pandas as pd
import os

data_dir = r"C:\Users\IZEROUAL\Desktop\MSPR3\data"

for filename in os.listdir(data_dir):
    if filename.endswith(".xls") or filename.endswith(".xlsx"):
        xls_path = os.path.join(data_dir, filename)
        csv_name = filename.replace(".xlsx", ".csv").replace(".xls", ".csv")
        csv_path = os.path.join(data_dir, csv_name)

        print(f"Conversion : {filename}...")
        try:
            # Essai 1 : lire comme CSV avec separateur ;
            df = pd.read_csv(xls_path, sep=";", encoding="latin-1")
            print("  -> Lu comme CSV (;)")
        except Exception:
            try:
                # Essai 2 : separateur tabulation
                df = pd.read_csv(xls_path, sep="\t", encoding="latin-1")
                print("  -> Lu comme CSV (tabulation)")
            except Exception as e2:
                print(f"  ERREUR : {e2}")
                continue

        print(f"  -> {len(df)} lignes, {len(df.columns)} colonnes")
        print(f"  -> Colonnes : {df.columns.tolist()[:8]}...")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  OK : {csv_name}")

print("Termine !")
