from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

class TopicModel:
    def __init__(self, embedding_model_name="indobenchmark/indobert-base-p1"):
        print(f"🔧 Inisialisasi Komponen BERTopic...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        self.hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        self.vectorizer_model = CountVectorizer(ngram_range=(1, 2))
        
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            language="indonesian",
            calculate_probabilities=True,
            verbose=True
        )

    def run_modeling(self, docs):
        print("\n🚀 Mulai Ekstrak Topik dengan BERTopic...")
        topics, probabilities = self.topic_model.fit_transform(docs)
        
        print("\n=== TOP 10 TOPIK DITEMUKAN ===")
        print(self.topic_model.get_topic_info().head(10))
        
        print("\n📊 Membuat dan Menyimpan Visualisasi HTML...")
        self.topic_model.visualize_barchart(top_n_topics=8).write_html("bertopic_barchart.html")
        self.topic_model.visualize_topics().write_html("bertopic_intertopic_map.html")
        self.topic_model.visualize_hierarchy().write_html("bertopic_hierarchy.html")
        print("✅ Visualisasi BERTopic Selesai!\n")
        
        return self.topic_model