import re
import json

# Unwanted patterns (same as in section_parser.py)
unwanted_patterns = [
    r"THE GAZETTE OF INDIA EXTRAORDINARY",
    r"PART II—SECTION 1",
    r"PUBLISHED BY AUTHORITY",
    r"MINISTRY OF LAW AND JUSTICE",
    r"Legislative Department",
    r"CORRIGENDA",
    r"NOTIFICATION",
]

def clean_text(text):
    for pattern in unwanted_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # normalize spacing
    return text.strip()

def clean_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        if "text" in entry:
            entry["text"] = clean_text(entry["text"])
        entry["plain_summary"] = ""  # Placeholder for plain-language summary
        entry["examples"] = []       # Placeholder for examples (list of strings)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_json = "consumer_protection_act_sections.json"
    output_json = "consumer_protection_act_sections_clean.json"

    clean_json(input_json, output_json)
    print(f"✅ Cleaned JSON saved to '{output_json}'")
