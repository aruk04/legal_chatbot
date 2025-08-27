from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from retriever import load_artifacts, retrieve_sections
from rule_engine import determine_forum, check_eligibility
from llm_agent import initialize_summarizer, generate_layman_summary
import joblib
from complaint_analyzer import ComplaintAnalyzer
import json
# New import for the intent prediction logic
from intent_predictor import load_predictor_models, predict_intent_with_fallback

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from your React frontend
# Adjust `allow_origins` in production to your frontend's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend runs on port 3000 by default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded artifacts and model
index = None
sections_data = None
model = None

# Global variable for ComplaintAnalyzer
complaint_analyzer = None

# Pydantic model for incoming chat requests
class ChatRequest(BaseModel):
    user_query: str
    price: str | None = None
    date: str | None = None

# Pydantic model for incoming complaint analysis requests
class ComplaintAnalysisRequest(BaseModel):
    complaint_text: str

class AddComplaintRequest(BaseModel):
    complaint_text: str
    intent_label: str

@app.on_event("startup")
def startup_event():
    global index, sections_data, model, complaint_analyzer
    print("Application startup: Loading artifacts and initializing LLM summarizer...")
    try:
        # Initialize LLM summarizer first (downloads model if not present)
        initialize_summarizer()
        
        # Load FAISS index and sections data
        faiss_index_path = "cpa_faiss_index.bin"
        sections_map_path = "sections_map.json"
        index, sections_data, model = load_artifacts(faiss_index_path, sections_map_path)
        print("✅ Backend artifacts loaded.")

        # Initialize and load ComplaintAnalyzer models for retraining/admin purposes
        complaint_analyzer = ComplaintAnalyzer()
        # We don't need to load classifier/hdbscan models here if intent_predictor handles it for inference
        # but it's needed for the add_complaint endpoint's retraining flow.
        complaint_analyzer.load_models(
            classifier_path='intent_classifier.joblib',
            hdbscan_path='hdbscan_model.joblib'
        )
        print("✅ Complaint analysis models (for retraining) loaded.")

        # Load models for the new intent_predictor
        load_predictor_models(
            classifier_path='intent_classifier.joblib',
            hdbscan_path='hdbscan_model.joblib'
        )
        print("✅ Intent Predictor models loaded.")

    except Exception as e:
        print(f"❌ Failed to load artifacts or initialize LLM: {e}")
        # Depending on severity, you might want to exit or disable LLM features
        raise HTTPException(status_code=500, detail="Failed to load necessary backend components.")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_query = request.user_query
        price = request.price
        date = request.date

        # Predict intent of the issue using the new fallback mechanism
        intent_prediction_result = predict_intent_with_fallback(user_query)

        # Combine user query with validated price and date for retrieval context
        user_query_with_context = user_query
        if price:
            user_query_with_context += f" price:{price}"
        if date:
            user_query_with_context += f" date:{date}"

        # Retrieve top relevant sections
        top_sections = retrieve_sections(user_query_with_context, index, sections_data, model, k=3)

        # Process sections: use plain_summary or generate with LLM
        processed_sections = []
        for sec in top_sections:
            display_text = sec['plain_summary']
            if not display_text: # If no plain_summary, use LLM to generate one
                display_text = generate_layman_summary(sec['text'], user_query)
            
            processed_sections.append({
                "section_id": sec['section_id'],
                "chapter": sec['chapter'],
                "display_text": display_text, # This will be either plain_summary or LLM generated
                "original_text_excerpt": sec['text'][:500] + "..." if len(sec['text']) > 500 else sec['text'], # Original for reference
                "examples": sec['examples'],
                "distance": sec['distance']
            })

        # Apply rule engine
        forum_recommendation = determine_forum(price)
        eligibility_status = check_eligibility(user_query, price, date, top_sections)

        return {
            "query": user_query,
            "predicted_intent": intent_prediction_result["intent"], # Use the intent from the new predictor
            "intent_details": intent_prediction_result, # Add full details for debugging/display
            "price": price,
            "date": date,
            "relevant_sections": processed_sections,
            "rule_engine_analysis": {
                "recommended_forum": forum_recommendation,
                "eligibility_status": eligibility_status
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# New endpoint for standalone complaint analysis (optional, but useful for admin)
@app.post("/analyze_complaint")
async def analyze_complaint(request: ComplaintAnalysisRequest):
    # Use the new intent predictor for analysis
    intent_prediction_result = predict_intent_with_fallback(request.complaint_text)
    
    return {"complaint_text": request.complaint_text, "predicted_intent_details": intent_prediction_result}

@app.post("/add_complaint")
async def add_complaint_data(request: AddComplaintRequest):
    if complaint_analyzer is None:
        raise HTTPException(status_code=500, detail="Complaint analyzer not initialized.")

    # Add the new complaint and label to the JSON file
    try:
        with open(complaint_analyzer.complaints_filepath, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            found = False
            for intent_data in data['intents']:
                if intent_data['tag'] == request.intent_label:
                    intent_data['patterns'].append(request.complaint_text)
                    found = True
                    break
            if not found:
                # If the intent label doesn't exist, create a new one
                data['intents'].append({"tag": request.intent_label, "patterns": [request.complaint_text], "responses": [request.intent_label]})
            f.seek(0)  # Rewind to the beginning of the file
            json.dump(data, f, indent=4)
            f.truncate() # Truncate any remaining parts of the old file
        
        # Trigger retraining
        complaint_analyzer.retrain_models()

        # Reload predictor models after retraining to pick up new mappings/models
        load_predictor_models( # Reload global models in intent_predictor
            classifier_path='intent_classifier.joblib',
            hdbscan_path='hdbscan_model.joblib'
        )

        return {"status": "success", "message": "Complaint added and models retrained.", "complaint_text": request.complaint_text, "intent_label": request.intent_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add complaint: {str(e)}")

if __name__ == "__main__":
    # To run the FastAPI server, use: uvicorn main:app --reload
    # The host should be 0.0.0.0 to be accessible from outside localhost for Docker/deployment
    uvicorn.run(app, host="0.0.0.0", port=8000)
