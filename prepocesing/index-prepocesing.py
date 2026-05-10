from func_prepocesing import preprocess, load_and_clean

path ="Top pariwisata Bangkalan.csv"

print("skrip jalan bg")
df = load_and_clean(path)
df["clean_text"] = df["text"].apply(preprocess)
df = df.drop(columns=["text"]) 
df.to_csv("Top Pariwisata Bangkalan (Cleaning with stopword).csv", index=False)
print("Done Preprocessingnya bang")

