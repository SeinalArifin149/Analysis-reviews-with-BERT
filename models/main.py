import pandas as pd
from sentiment_model import SentimentModel
from bertopic_models import TopicModel

if __name__ == "__main__":
    file_path = "../prepocesing/Top Pariwisata Bangkalan (Cleaning).csv"
    # file_path = "Top Pariwisata Bangkalan (Cleaning with stopword).csv"

    try:
        df = pd.read_csv(file_path)
        df["teks"] = df["teks"].astype(str)
        df = df.dropna(subset=["teks", "stars"])
        df["teks"] = df["teks"].astype(str)
        docs = df["teks"].tolist()

    except FileNotFoundError:
        print("Pile tidak ditemukan")
        exit()
    
    Sentiment_Modeler =SentimentModel()

    Sentiment_Modeler.train(df)
    
    # iseng aja tambah uji coba 

    test_text = "akses ke tempat wisata ini sudah sangat baik"
    hasil_prediksi = Sentiment_Modeler.predict(test_text)
    print(f"\nUji Prediksi: '{test_text}' --> {hasil_prediksi}\n")

    topic_modeler = TopicModel()
    model_bertopic = topic_modeler.run_modeling(docs)

df["hasil_sentimen"] = df["teks"].apply(Sentiment_Modeler.predict)
df["hasil_topik"] = model_bertopic.topics_

name_file = "Top Pariwisata Bangkalan (Train).csv"
df.to_csv(name_file, index=False)