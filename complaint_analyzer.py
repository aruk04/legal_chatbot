import json
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import hdbscan
import numpy as np
import joblib

class ComplaintAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2', complaints_filepath='complaints_data.json'):
        self.model = SentenceTransformer(model_name)
        self.complaints_filepath = complaints_filepath
        self.intents = []
        self.intent_classifier = None
        self.hdbscan_model = None
        self.complaint_embeddings = None
        self.complaint_texts = []
        self.complaint_labels = []

    def load_complaints(self):
        with open(self.complaints_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for intent_data in data['intents']:
            tag = intent_data['tag']
            for pattern in intent_data['patterns']:
                self.complaint_texts.append(pattern)
                self.complaint_labels.append(tag)
        self.intents = sorted(list(set(self.complaint_labels)))
        print(f"✅ Loaded {len(self.complaint_texts)} complaints and {len(self.intents)} unique intents.")

    def generate_embeddings(self):
        print("Generating embeddings for complaints...")
        self.complaint_embeddings = self.model.encode(self.complaint_texts, show_progress_bar=True)
        print("✅ Embeddings generated.")

    def train_intent_classifier(self):
        if self.complaint_embeddings is None or len(self.complaint_labels) == 0:
            print("❌ No complaint data or embeddings to train the classifier. Please load complaints and generate embeddings first.")
            return
        
        print("Training intent classifier...")
        
        # Check if stratification is possible for all classes
        min_samples_per_class = min(self.complaint_labels.count(label) for label in set(self.complaint_labels))
        
        if min_samples_per_class < 2 and len(set(self.complaint_labels)) > 1: # Cannot stratify if any class has only 1 sample, and there's more than one class
            print("⚠️ Warning: Cannot stratify train_test_split as at least one class has fewer than 2 samples. Splitting without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(
                self.complaint_embeddings, self.complaint_labels, test_size=0.5, random_state=42
            )
        elif len(set(self.complaint_labels)) == 1: # If only one class, no need to stratify
            print("⚠️ Warning: Only one class available. Splitting without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(
                self.complaint_embeddings, self.complaint_labels, test_size=0.5, random_state=42
            )
        else:
            # Normal stratification
            X_train, X_test, y_train, y_test = train_test_split(
                self.complaint_embeddings, self.complaint_labels, test_size=0.5, random_state=42, stratify=self.complaint_labels
            )

        # Ensure X_train and y_train are not empty before fitting
        if len(X_train) == 0 or len(y_train) == 0:
            print("❌ Training data is empty after split. Cannot train classifier.")
            self.intent_classifier = None # Ensure classifier is not partially trained
            return

        self.intent_classifier = LogisticRegression(random_state=42, solver='liblinear', multi_class='auto', max_iter=1000)
        self.intent_classifier.fit(X_train, y_train)

        y_pred = self.intent_classifier.predict(X_test)
        print("✅ Intent Classifier trained.")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    def predict_intent(self, text):
        if self.intent_classifier is None:
            print("❌ Intent classifier not trained. Please train it first.")
            return "Unknown Intent"
        
        embedding = self.model.encode([text])
        prediction = self.intent_classifier.predict(embedding)[0]
        return prediction

    def perform_hdbscan_clustering(self):
        if self.complaint_embeddings is None:
            print("❌ No complaint embeddings to perform clustering. Please generate embeddings first.")
            return
        
        print("Performing HDBSCAN clustering...")
        # min_cluster_size and min_samples are key parameters to tune
        # min_cluster_size: The smallest size grouping that you wish to consider a cluster
        # min_samples: Controls how conservative the clustering is. Larger values mean more points are considered noise.
        self.hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=None, prediction_data=True)
        cluster_labels = self.hdbscan_model.fit_predict(self.complaint_embeddings)
        
        self.cluster_labels = cluster_labels
        print("✅ HDBSCAN Clustering complete.")
        print(f"Found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters. {np.sum(cluster_labels == -1)} points identified as noise.")
        
        # Optional: Print samples from each cluster
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == -1:
                print(f"\nNoise Samples ({np.sum(cluster_labels == -1)} points):")
            else:
                print(f"\nCluster {cluster_id} Samples:")
            
            cluster_samples = [self.complaint_texts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            for sample in cluster_samples[:3]: # Print up to 3 samples per cluster
                print(f"  - {sample}")

    def retrain_models(self):
        print("Initiating model retraining...")
        self.complaint_texts = []
        self.complaint_labels = []
        self.load_complaints()
        self.generate_embeddings()
        self.train_intent_classifier()
        self.perform_hdbscan_clustering()
        self.save_models()
        print("✅ Model retraining complete.")

    def get_noise_samples(self):
        if self.complaint_embeddings is None or not hasattr(self, 'cluster_labels'):
            return []
        noise_indices = np.where(self.cluster_labels == -1)[0]
        return [self.complaint_texts[i] for i in noise_indices]

    def get_cluster_samples(self, cluster_id, num_samples=3):
        if self.complaint_embeddings is None or not hasattr(self, 'cluster_labels'):
            return []
        cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
        return [self.complaint_texts[i] for i in cluster_indices[:num_samples]]

    def get_all_clusters_with_samples(self, num_samples=3):
        if self.complaint_embeddings is None or not hasattr(self, 'cluster_labels'):
            return {}

        cluster_info = {}
        unique_clusters = sorted(set(self.cluster_labels))

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                cluster_info["noise"] = self.get_noise_samples()
            else:
                cluster_info[f"cluster_{cluster_id}"] = self.get_cluster_samples(cluster_id, num_samples)
        return cluster_info

    def save_models(self, classifier_path='intent_classifier.joblib', hdbscan_path='hdbscan_model.joblib'):
        if self.intent_classifier:
            joblib.dump(self.intent_classifier, classifier_path)
            print(f"✅ Intent classifier saved to {classifier_path}")
        if self.hdbscan_model:
            joblib.dump(self.hdbscan_model, hdbscan_path)
            print(f"✅ HDBSCAN model saved to {hdbscan_path}")

    def load_models(self, classifier_path='intent_classifier.joblib', hdbscan_path='hdbscan_model.joblib'):
        try:
            self.intent_classifier = joblib.load(classifier_path)
            print(f"✅ Intent classifier loaded from {classifier_path}")
        except FileNotFoundError:
            print(f"❌ Intent classifier model not found at {classifier_path}")
        
        try:
            self.hdbscan_model = joblib.load(hdbscan_path)
            print(f"✅ HDBSCAN model loaded from {hdbscan_path}")
        except FileNotFoundError:
            print(f"❌ HDBSCAN model not found at {hdbscan_path}")

if __name__ == "__main__":
    analyzer = ComplaintAnalyzer()
    analyzer.load_complaints()
    analyzer.generate_embeddings()
    analyzer.train_intent_classifier()
    analyzer.perform_hdbscan_clustering()
    analyzer.save_models()

    # Example usage after training/loading
    test_complaint = "My new TV has a blank screen and won't turn on after a week."
    predicted_intent = analyzer.predict_intent(test_complaint)
    print(f"\nTest Complaint: \"{test_complaint}\"\nPredicted Intent: {predicted_intent}")

    test_complaint_2 = "The delivery service lost my package and now they are not responding."
    predicted_intent_2 = analyzer.predict_intent(test_complaint_2)
    print(f"\nTest Complaint: \"{test_complaint_2}\"\nPredicted Intent: {predicted_intent_2}")

    test_complaint_3 = "I got a bad haircut and now my hair is ruined."
    predicted_intent_3 = analyzer.predict_intent(test_complaint_3)
    print(f"\nTest Complaint: \"{test_complaint_3}\"\nPredicted Intent: {predicted_intent_3}")
