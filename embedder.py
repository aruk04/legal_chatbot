import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def create_embeddings_and_index(sections_filepath, model_name='all-MiniLM-L6-v2'):
    """
    Loads sections, creates embeddings, and builds a FAISS index.
    """
    # Load sections
    with open(sections_filepath, 'r', encoding='utf-8') as f:
        sections = json.load(f)

    # Extract texts for embedding
    section_texts = [section['text'] for section in sections]
    section_ids = [section['section_id'] for section in sections]
    section_chapters = [section['chapter'] for section in sections]

    # Load SBERT model
    model = SentenceTransformer(model_name)

    # Generate embeddings
    print(f"Generating embeddings for {len(section_texts)} sections...")
    embeddings = model.encode(section_texts, show_progress_bar=True)
    print("Embeddings generated.")

    # Convert to numpy array with float32 for FAISS
    embeddings = np.array(embeddings).astype('float32')

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    index.add(embeddings)
    print(f"FAISS index created with {index.ntotal} vectors.")

    return index, sections, model

def save_artifacts(index, sections, output_index_path, output_sections_map_path):
    """
    Saves the FAISS index and the sections data.
    """
    faiss.write_index(index, output_index_path)
    with open(output_sections_map_path, 'w', encoding='utf-8') as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)
    print(f"FAISS index saved to '{output_index_path}'")
    print(f"Section map saved to '{output_sections_map_path}'")

if __name__ == "__main__":
    input_sections_filepath = "consumer_protection_act_sections_clean.json"
    output_faiss_index_path = "cpa_faiss_index.bin"
    output_sections_map_path = "sections_map.json"

    # Update todo status
    print("Creating embeddings and FAISS index...")
    
    index, sections, model = create_embeddings_and_index(input_sections_filepath)
    save_artifacts(index, sections, output_faiss_index_path, output_sections_map_path)

    print("Embeddings and FAISS index creation complete.")
