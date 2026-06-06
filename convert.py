import pandas as pd

df = pd.read_excel(r'C:\Users\IZEROUAL\Desktop\MSPR3\data\eCO2mix_RTE_Annuel-Definitif_2020.xls', sheet_name=0)
print('Colonnes:', df.columns.tolist())
print('Lignes:', len(df))
print(df.head(3))
df.to_csv(r'C:\Users\IZEROUAL\Desktop\MSPR3\data\eco2mix_2020.csv', index=False, encoding='utf-8-sig')
print('CSV créé !')
