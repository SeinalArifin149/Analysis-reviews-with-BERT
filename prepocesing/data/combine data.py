import pandas as pd

# 1. Membaca semua dataset
df1 = pd.read_csv("Aermata Arosbaya.csv")
df2 = pd.read_csv("Bukit Jaddih.csv")
df3 = pd.read_csv("Masjid Syaikhona.csv")
df4 = pd.read_csv("Stadion Glora.csv")
df5 = pd.read_csv("Taman Mangrove sepulu.csv")

dataframes = [df1, df2, df3, df4, df5]
for df in dataframes:
    df.columns = df.columns.str.lower().str.strip()

df_gabungan = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

print("--- Data Berhasil Digabung ---")
print("Dimensi data baru:", df_gabungan.shape)


df_gabungan.to_csv("Top pariwisata Bangkalan.csv", index=False)

print("\nMantap! Kelima data wisata Bangkalan sudah bersatu dan tersimpan sebagai 'Top Pariwisata Bangkalan.csv'")