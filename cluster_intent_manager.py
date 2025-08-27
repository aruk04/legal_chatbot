import json
import os

MAPPING_FILE = 'cluster_to_intent_map.json'
REVIEW_FILE = 'pending_review.jsonl' # JSON Lines format for easy appending

def load_mappings():
    """Loads existing cluster ID to intent name mappings."""
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_mappings(mappings):
    """Saves the current cluster ID to intent name mappings."""
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=4)
    print(f"✅ Cluster to intent mappings saved to {MAPPING_FILE}")

def map_cluster_to_intent(cluster_id, intent_name):
    """Maps a specific HDBSCAN cluster ID to a human-readable intent name."""
    mappings = load_mappings()
    mappings[str(cluster_id)] = intent_name
    save_mappings(mappings)
    print(f"✅ Mapped cluster {cluster_id} to intent '{intent_name}'")

def log_for_review(complaint_text, cluster_label, cluster_strength):
    """Logs a complaint for human review, especially for new or noisy clusters."""
    review_entry = {
        "complaint_text": complaint_text,
        "cluster_label": str(cluster_label),
        "cluster_strength": float(cluster_strength) if cluster_strength is not None else None,
        "timestamp": json.dumps(os.times().elapsed), # Simple timestamp
        "status": "pending"
    }
    with open(REVIEW_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(review_entry) + '\n')
    print(f"✅ Complaint logged for review (Cluster: {cluster_label}).")

def get_pending_reviews():
    """Reads all complaints currently pending review."""
    reviews = []
    if os.path.exists(REVIEW_FILE):
        with open(REVIEW_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                reviews.append(json.loads(line))
    return reviews

def clear_pending_reviews():
    """Clears the pending review log file."""
    if os.path.exists(REVIEW_FILE):
        os.remove(REVIEW_FILE)
        print(f"✅ {REVIEW_FILE} cleared.")
    else:
        print(f"❌ {REVIEW_FILE} does not exist.")

if __name__ == "__main__":
    # Example Usage:
    print("--- Testing Cluster Intent Manager ---")

    # Clear previous data
    clear_pending_reviews()
    if os.path.exists(MAPPING_FILE):
        os.remove(MAPPING_FILE)
        print(f"✅ Cleaned up {MAPPING_FILE}.")

    # Log some complaints for review
    log_for_review("My phone is making strange crackling sounds.", -1, 0.1)
    log_for_review("The touchscreen is completely frozen after update.", 2, 0.8)
    log_for_review("This new feature keeps crashing the app.", 5, 0.6)

    print("\nPending Reviews:")
    for review in get_pending_reviews():
        print(review)

    # Map a cluster
    map_cluster_to_intent(2, "touchscreen issue")
    map_cluster_to_intent(5, "software bug")

    print("\nUpdated Mappings:")
    print(load_mappings())

    # Log another complaint for review
    log_for_review("My data is draining too fast.", -1, 0.3)

    print("\nPending Reviews after more logging:")
    for review in get_pending_reviews():
        print(review)

    print("\nMapping cluster 2 to 'display issue' (overwriting previous)")
    map_cluster_to_intent(2, "display issue")
    print("\nUpdated Mappings:")
    print(load_mappings())
