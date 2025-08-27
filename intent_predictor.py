import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
import cluster_intent_manager # We'll keep this for mapped clusters, but remove logging new ones
import json # For loading complaints_data.json
import os

# Global variables
sentence_model = None
intent_classifier = None
hdbscan_model = None
cluster_to_intent_map = {}

# New global variables for similarity search fallback
all_known_complaint_patterns = []
all_known_complaint_labels = []
all_known_complaint_embeddings = None

# Configurable threshold
CONFIDENCE_THRESHOLD = 0.65  # Adjust dynamically if needed
SIMILARITY_THRESHOLD = 0.5   # Threshold for fallback similarity, adjust as needed
COMPLAINTS_FILEPATH = 'complaints_data.json' # Define complaints file path

def load_predictor_models(model_name='all-MiniLM-L6-v2',
                          classifier_path='intent_classifier.joblib',
                          hdbscan_path='hdbscan_model.joblib'):
    """Load SentenceTransformer, LR classifier, HDBSCAN, and mappings, and all complaint data."""
    global sentence_model, intent_classifier, hdbscan_model, cluster_to_intent_map
    global all_known_complaint_patterns, all_known_complaint_labels, all_known_complaint_embeddings

    print("Loading intent prediction models and data...")

    sentence_model = SentenceTransformer(model_name)
    intent_classifier = joblib.load(classifier_path)
    hdbscan_model = joblib.load(hdbscan_path)
    cluster_to_intent_map = cluster_intent_manager.load_mappings()

    # Load all known complaints for similarity fallback
    if os.path.exists(COMPLAINTS_FILEPATH):
        with open(COMPLAINTS_FILEPATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for intent_data in data['intents']:
            tag = intent_data['tag']
            for pattern in intent_data['patterns']:
                all_known_complaint_patterns.append(pattern)
                all_known_complaint_labels.append(tag)
        all_known_complaint_embeddings = sentence_model.encode(all_known_complaint_patterns, show_progress_bar=False)
        print(f"✅ Loaded {len(all_known_complaint_patterns)} known complaint patterns for fallback.")
    else:
        print(f"❌ Complaints data file not found at {COMPLAINTS_FILEPATH}. Similarity fallback will be limited.")

    print("✅ Models and mappings loaded for intent prediction.")

def predict_intent_with_fallback(text: str) -> dict:
    """Predict intent using LR → fallback to HDBSCAN → fallback to similarity search."""

    if sentence_model is None:
        return {"intent": "system_error", "confidence": 0.0, "source": "none"}

    # 1. Embed text
    embedding = sentence_model.encode([text])

    # 2. Logistic Regression first
    if intent_classifier:
        probabilities = intent_classifier.predict_proba(embedding)[0]
        max_prob = np.max(probabilities)
        predicted_class_index = np.argmax(probabilities)
        predicted_intent_lr = intent_classifier.classes_[predicted_class_index]

        if max_prob >= CONFIDENCE_THRESHOLD:
            return {
                "intent": predicted_intent_lr,
                "confidence": float(max_prob),
                "source": "LogisticRegression",
                "cluster_label": None,
                "cluster_strength": None # Add cluster_strength for consistency in output
            }
        else:
            print(f"⚠ Low confidence ({max_prob:.2f}) from Logistic Regression. Falling back to HDBSCAN.")

    # 3. HDBSCAN fallback
    if hdbscan_model and all_known_complaint_embeddings is not None:
        cluster_labels, cluster_strengths = hdbscan.prediction.approximate_predict(hdbscan_model, embedding)
        cluster_label = cluster_labels[0]
        cluster_strength = cluster_strengths[0]

        if cluster_label != -1:  # Found a cluster
            mapped_intent = cluster_to_intent_map.get(str(cluster_label))
            if mapped_intent:
                return {
                    "intent": mapped_intent,
                    "confidence": float(cluster_strength),
                    "source": "HDBSCAN_Mapped",
                    "cluster_label": int(cluster_label),
                    "cluster_strength": float(cluster_strength)
                }
            else:
                # New cluster discovered by HDBSCAN, but no manual mapping. Fallback to similarity search.
                print(f"⚠ New cluster ({cluster_label}) discovered by HDBSCAN, but no mapping. Falling back to similarity search.")
                pass # Continue to similarity fallback
        else:
            # HDBSCAN identified as noise. Fallback to similarity search.
            print("⚠ HDBSCAN identified as noise. Falling back to similarity search.")
            pass # Continue to similarity fallback

    # 4. Final fallback: Similarity search against all known complaint patterns
    if all_known_complaint_embeddings is not None and len(all_known_complaint_embeddings) > 0:
        # Calculate cosine similarity between the query embedding and all known complaint embeddings
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embedding, all_known_complaint_embeddings)[0]
        
        max_similarity_index = np.argmax(similarities)
        max_similarity = similarities[max_similarity_index]
        
        if max_similarity >= SIMILARITY_THRESHOLD:
            predicted_intent_fallback = all_known_complaint_labels[max_similarity_index]
            return {
                "intent": predicted_intent_fallback,
                "confidence": float(max_similarity),
                "source": "Similarity_Fallback",
                "cluster_label": None,
                "cluster_strength": None,
                "message": "Classified via similarity to known patterns."
            }
        else:
            return {"intent": "unclassified", "confidence": float(max_similarity), "source": "Similarity_Fallback", "cluster_label": None, "cluster_strength": None, "message": "No confident classification found, even with similarity fallback."}

    # If no models or fallback mechanisms can provide a definitive answer
    return {"intent": "unclassified", "confidence": 0.0, "source": "None_Available", "cluster_label": None, "cluster_strength": None, "message": "No classification models or fallback mechanisms are available."}


if __name__ == "__main__":
    # Example usage for testing
    # Ensure complaint_analyzer.py has been run at least once to create joblib files
    load_predictor_models()

    print("\n--- Testing Intent Predictor with Full Fallback ---")

    # Test 1: High confidence LR prediction (e.g., battery issue)
    print("\nTest Case 1: High confidence LR prediction")
    result1 = predict_intent_with_fallback("My phone battery drains very quickly, even with light use.")
    print(result1)

    # Test 2: Low confidence LR, HDBSCAN maps to known cluster (requires mapping in cluster_to_intent_map.json)
    # To test this, you might manually create cluster_to_intent_map.json like {"2": "touchscreen issue"}
    print("\nTest Case 2: Low confidence LR, HDBSCAN maps to known cluster (mocked mapping for cluster 2)")
    # Make sure to run cluster_intent_manager.py to create the map file or manually create it for this test
    cluster_intent_manager.map_cluster_to_intent(2, "touchscreen issue")
    result2 = predict_intent_with_fallback("My phone screen is completely unresponsive to touch in the top left corner.")
    print(result2)

    # Test 3: Low confidence LR, HDBSCAN new cluster/noise -> Similarity Fallback (e.g., new 'camera issue')
    print("\nTest Case 3: Low confidence LR, HDBSCAN new cluster/noise -> Similarity Fallback (camera issue)")
    # This should now fall back to similarity search and find 'camera issue' if it's in complaints_data.json
    result3 = predict_intent_with_fallback("My phone's camera suddenly stopped working after the latest software update and shows a black screen.")
    print(result3)

    # Test 4: Low confidence LR, HDBSCAN noise -> Similarity Fallback (e.g., 'refund issue' but very ambiguous)
    print("\nTest Case 4: Low confidence LR, HDBSCAN noise -> Similarity Fallback (refund issue, ambiguous)")
    result4 = predict_intent_with_fallback("I did not get my money back from the shop for the faulty item.")
    print(result4)

    # Test 5: Truly unclassified even by similarity (e.g., a very niche or irrelevant query)
    print("\nTest Case 5: Truly unclassified (very niche query)")
    result5 = predict_intent_with_fallback("What is the capital of France?")
    print(result5)
