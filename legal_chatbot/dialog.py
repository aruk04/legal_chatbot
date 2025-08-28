# # eligibility_dialog.py
# from typing import Optional, Dict
# from legal_chatbot.eligibility_classifier import predict_label, predict_proba, load_model

# # Confidence band where we ask more qs
# LOW, HIGH = 0.40, 0.80

# def next_question(state: Dict, proba: Optional[float]) -> Optional[str]:
#     """
#     Return the most informative next question based on missing fields and uncertainty.
#     Order of importance:
#       1) purchase_date  (limitation)
#       2) use_context    (consumer vs resale/b2b vs livelihood)
#       3) warranty_period_months (if defect-like intents)
#       4) complaint_value
#       5) attempts_to_resolve / current_status if still uncertain
#     """
#     intent = (state.get("intent") or "unknown").lower()
#     defect_like = intent in {"screen_issue","battery_issue","performance_issue","warranty_issue","replacement_denial"}

#     if not state.get("purchase_date"):
#         return "When did you purchase the product? (YYYY-MM-DD)"
#     if not state.get("use_context"):
#         return "Was the product for personal use, self-employment livelihood, or resale/B2B?"
#     if defect_like and state.get("warranty_period_months") is None:
#         return "What is/was the warranty period in months? (0 if none)"
#     if state.get("complaint_value") is None:
#         return "What is the product price or claim value (in INR)?"
#     # If still uncertain ask about resolution attempts & status
#     if proba is not None and LOW <= proba <= HIGH:
#         if state.get("attempts_to_resolve") is None:
#             return "Have you contacted the seller/service center? Do you have emails/chats or job-sheets?"
#         if state.get("current_status") is None:
#             return "Did you already receive a full refund or a replacement?"
#     return None

# class EligibilityFlow:
#     def __init__(self, model=None, threshold: float = 0.5, max_q: int = 3):
#         self.model = model or load_model()
#         self.threshold = threshold
#         self.max_q = max_q

#     def step(self, state: Dict) -> Dict:
#         """
#         One turn of the dialog:
#           - score with current info
#           - decide next question (if any)
#           - stop when confident or max questions reached
#         """
#         asked = state.get("_questions_asked", 0)
#         proba = predict_proba(self.model, state)
#         q = next_question(state, proba)

#         # stop if confident or hit question limit or nothing left to ask
#         confident = (proba >= HIGH) or (proba <= (1 - HIGH))
#         done = confident or (q is None) or (asked >= self.max_q)

#         result = predict_label(self.model, state, threshold=self.threshold)

#         return {
#             "done": done,
#             "next_question": (None if done else q),
#             "proba": proba,
#             "result": result
#         }


# eligibility_dialog.py
from typing import Optional, Dict
from eligibility_classifier import predict_label, predict_proba, load_model

# You can tighten/loosen this band; the main fix is to NOT finish while required slots are missing.
LOW, HIGH = 0.40, 0.80

# Declare the core slots that must be asked at least once
CORE_SLOTS = ("purchase_date", "use_context")
# Optional slots asked for defect-like issues
DEFECT_SLOTS = ("warranty_period_months", )
# General helpful slot
VALUE_SLOT = "complaint_value"

def _core_slots_missing(state: Dict, intent: str) -> bool:
    # Must always collect purchase_date and use_context
    if not state.get("purchase_date"):
        return True
    if not state.get("use_context"):
        return True

    # If defect-like, also try to collect warranty months
    defect_like = intent in {"screen_issue","battery_issue","performance_issue","warranty_issue","replacement_denial"}
    if defect_like and (state.get("warranty_period_months") is None):
        return True

    # Value is useful for forum & downstream; not strictly required to stop
    return False

def next_question(state: Dict, proba: Optional[float]) -> Optional[str]:
    intent = (state.get("intent") or "unknown").lower()
    defect_like = intent in {"screen_issue","battery_issue","performance_issue","warranty_issue","replacement_denial"}

    # Ask core slots first, always
    if not state.get("purchase_date"):
        return "When did you purchase the product? (YYYY-MM-DD)"
    if not state.get("use_context"):
        return "Was the product for personal use, self-employment livelihood, or resale/B2B?"

    # Then defect-specific
    if defect_like and state.get("warranty_period_months") is None:
        return "What is/was the warranty period in months? (0 if none)"

    # Then value if still missing
    if state.get("complaint_value") is None:
        return "What is the product price or claim value (in INR)?"

    # If still uncertain, ask helpful follow-ups
    if proba is not None and LOW <= proba <= HIGH:
        if state.get("attempts_to_resolve") is None:
            return "Have you contacted the seller/service center? Do you have emails/chats or job-sheets?"
        if state.get("current_status") is None:
            return "Did you already receive a full refund or a replacement?"

    return None

class EligibilityFlow:
    def _init_(self, model=None, threshold: float | None = None, max_q: int = 3):
        # model is the blob from eligibility_classifier.load_model()
        self.model = model or load_model()
        self.threshold = threshold  # if None, predict_label uses the tuned threshold saved with the model
        self.max_q = max_q

    def step(self, state: Dict) -> Dict:
        asked = state.get("_questions_asked", 0)
        intent = (state.get("intent") or "unknown").lower()

        # Score with current info
        proba = predict_proba(self.model, state)
        # Decide what to ask next
        q = next_question(state, proba)

        # --- KEY CHANGE ---
        # Do not finish while core slots are missing, even if the model is confident.
        core_missing = _core_slots_missing(state, intent)

        # Only finish when:
        #  - no core slots are missing, AND
        #  - (we're confident OR there's nothing left to ask OR we've hit the max questions)
        confident = (proba >= HIGH) or (proba <= (1 - HIGH))
        done = (not core_missing) and (confident or (q is None) or (asked >= self.max_q))

        result = predict_label(self.model, state, threshold=self.threshold)

        return {
            "done": done,
            "next_question": (None if done else q),
            "proba": proba,
            "result": result
        }
