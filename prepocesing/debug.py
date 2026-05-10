import pandas as pd

df = pd.read_csv("Top pariwisata Bangkalan.csv")

# 1. Cek missing value (nilai kosong) di setiap kolom
print("--- Jumlah Missing Value per Kolom ---")
print(df.isna().sum())

print ("---Ini dimensi data")
print(df.shape)

# 2. Cek jumlah baris data yang duplikat (sama persis)
print("\n--- Jumlah Data Duplikat ---")
print(df.duplicated().sum())