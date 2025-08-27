import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from rule_engine import determine_forum, check_eligibility
from llm_agent import initialize_summarizer, generate_layman_summary

def load_artifacts(output_index_path, output_sections_map_path, model_name='all-MiniLM-L6-v2'):
    """
    Loads the FAISS index, the sections data, and the SBERT model.
    """
    index = faiss.read_index(output_index_path)
    with open(output_sections_map_path, 'r', encoding='utf-8') as f:
        sections = json.load(f)
    model = SentenceTransformer(model_name)
    print("Artifacts loaded successfully.")
    return index, sections, model

def retrieve_sections(query, index, sections, model, k=5):
    """
    Embeds the query and retrieves the top-k most relevant sections.
    """
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    # Search the FAISS index
    distances, indices = index.search(query_embedding, k) # D for distances, I for indices

    # Retrieve the actual sections
    retrieved_sections = []
    for i, idx in enumerate(indices[0]):
        section_info = sections[idx]
        retrieved_sections.append({
            "section_id": section_info['section_id'],
            "chapter": section_info['chapter'],
            "text": section_info['text'],
            "plain_summary": section_info.get('plain_summary', ''),  # Get summary, default to empty string
            "examples": section_info.get('examples', []),          # Get examples, default to empty list
            "distance": float(distances[0][i]) # Convert numpy float to Python float
        })
    return retrieved_sections

def get_validated_input(prompt, pattern, error_message, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        user_input = input(prompt)
        if re.fullmatch(pattern, user_input):
            return user_input
        else:
            print(error_message)
            attempts += 1
    print("Too many invalid attempts. Please restart the process.")
    return None

if __name__ == "__main__":
    faiss_index_path = "cpa_faiss_index.bin"
    sections_map_path = "sections_map.json"

    try:
        initialize_summarizer()
        index, sections_data, model = load_artifacts(faiss_index_path, sections_map_path)

        while True:
            user_query = input("\nEnter your legal query (or 'quit' to exit): ")
            if user_query.lower() == 'quit':
                break

            # Example of how to use the validated input for price and date
            price = get_validated_input("Enter price (numeric value, e.g., 34000): ", r"^\d+$", "Please provide the numeric value for the price (e.g., 34000).")
            if price is None:
                break

            date = get_validated_input("Enter date (DD-MM-YYYY, e.g., 14-08-2025): ", r"^(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-\d{4}$", "Please provide the date in DD-MM-YYYY format (e.g., 14-08-2025).")
            if date is None:
                break

            # You can then use 'price' and 'date' in your query or rule engine
            # For now, let's just append them to the query for demonstration
            user_query_with_context = f"{user_query} price:{price} date:{date}"
            
            top_sections = retrieve_sections(user_query_with_context, index, sections_data, model, k=3)

            print("\nTop 3 relevant sections:")
            for sec in top_sections:
                print(f"-- Section ID: {sec['section_id']}")
                print(f"   Chapter: {sec['chapter']}")
                if sec['plain_summary']:
                    print(f"   Text (summary): {sec['plain_summary']}")
                else:
                    # Generate summary using LLM if plain_summary is not available
                    llm_generated_summary = generate_layman_summary(sec['text'], user_query)
                    print(f"   Text (LLM Summary): {llm_generated_summary}")
                if sec['examples']:
                    print(f"   Examples: {', '.join(sec['examples'])}")
                print(f"   Distance: {sec['distance']:.4f}")
                print("---------------------")

            # Apply rule engine
            print("\n--- Rule Engine Analysis ---")
            forum_recommendation = determine_forum(price)
            eligibility_status = check_eligibility(user_query, price, date, top_sections)
            print(f"Recommended Forum: {forum_recommendation}")
            print(f"Eligibility Status:\n{eligibility_status}")
            print("----------------------------")

    except FileNotFoundError as e:
        print(f"Error: Required file not found. Make sure '{e.filename}' exists.")
    except Exception as e:
        print(f"An error occurred: {e}")
