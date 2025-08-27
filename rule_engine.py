import re

def determine_forum(complaint_value):
    """
    Determines the appropriate consumer forum based on the complaint value.
    """
    if complaint_value is None:
        return "Please provide a valid complaint value to determine the forum."
    
    try:
        value = int(complaint_value)
        if value <= 5000000:  # Up to 50 Lakhs
            return "District Commission"
        elif value <= 20000000:  # Up to 2 Crores
            return "State Commission"
        else:  # Above 2 Crores
            return "National Commission"
    except ValueError:
        return "Invalid complaint value. Please provide a numeric value."

def check_eligibility(user_query, price=None, date=None, retrieved_sections=None):
    """
    Checks general eligibility based on user query and other parameters.
    This is a placeholder and can be expanded with more complex rules.
    """
    eligibility_messages = []

    # Basic eligibility check based on presence of price and date (example)
    if price and date:
        eligibility_messages.append("Basic eligibility criteria (price and date) met.")
    else:
        eligibility_messages.append("Price and/or date information is missing, which may affect eligibility.")

    if retrieved_sections:
        # Example: check if any retrieved section directly mentions 'consumer rights' or 'deficiency'
        relevant_keywords = ["consumer rights", "deficiency", "unfair trade practice", "defective"]
        found_keywords = []
        for section in retrieved_sections:
            for keyword in relevant_keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', section['text'], re.IGNORECASE):
                    found_keywords.append(keyword)
        
        if found_keywords:
            eligibility_messages.append(f"Retrieved sections contain relevant keywords: {', '.join(set(found_keywords))}.")
        else:
            eligibility_messages.append("No direct keywords related to common consumer issues found in retrieved sections.")

    if not eligibility_messages:
        eligibility_messages.append("No specific eligibility rules triggered based on provided information.")

    return "\n".join(eligibility_messages)


if __name__ == "__main__":
    # Example Usage
    print("Forum for 40 Lakhs: ", determine_forum(4000000))
    print("Forum for 1.5 Crores: ", determine_forum(15000000))
    print("Forum for 5 Crores: ", determine_forum(50000000))
    print("Forum for invalid value: ", determine_forum("abc"))

    # Example eligibility check
    sample_sections = [
        {"text": "This section talks about consumer rights and their protection.", "section_id": "1"},
        {"text": "Definition of deficiency in service.", "section_id": "2"}
    ]
    print("\nEligibility check 1:")
    print(check_eligibility(user_query="My product is defective", price="50000", date="01-01-2023", retrieved_sections=sample_sections))

    print("\nEligibility check 2:")
    print(check_eligibility(user_query="General query", price=None, date=None, retrieved_sections=[]))
