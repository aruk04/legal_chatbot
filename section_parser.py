import re
import json
from pdfminer.high_level import extract_text

def parse_cpa_sections(text_content):
    """
    Parses the text content of the Consumer Protection Act, 2019 into sections.
    Each section will include its official text and associated chapter.
    """
    sections = []
    chapter_info = []

    # Match CHAPTER headings (Roman numerals + title)
    chapter_pattern = re.compile(r'(CHAPTER\s+([IVXLCDM]+)\s+([A-Z\s]+))', re.MULTILINE)
    for match in chapter_pattern.finditer(text_content):
        chapter_info.append({
            "start": match.start(),
            "title": match.group(3).strip()
        })

    # Match section starts (e.g., "1.", "2.", etc.)
    section_start_pattern = re.compile(r'^\s*(\d+)\.', re.MULTILINE)

    section_starts = []
    for match in section_start_pattern.finditer(text_content):
        section_starts.append({
            "index": match.start(),
            "number": match.group(1)
        })

    if not section_starts:
        print("No sections found with the current pattern.")
        return []

    # Clean unwanted text patterns
    unwanted_patterns = [
        r"THE GAZETTE OF INDIA\s*EXTRAORDINARY",
        r"PART\s*II\s*—\s*SECTION\s*1",
        r"PUBLISHED\s*BY\s*AUTHORITY",
        r"MINISTRY OF LAW AND JUSTICE",
        r"Legislative Department",
        r"New Delhi,\s*the\s*\d{1,2}\s*[A-Za-z]+\s*\d{4}",
        r"REGISTERED NO\.\s*[A-Z0-9\-()]+",
        r"\bCORRIGENDA\b",
        r"\bNOTIFICATION\b",
        r"\bExtraordinary\b",
        r"\n\s*\d{1,3}\s*\n",  # page numbers
        r"bl Hkkx esa fHkUu i\"B la\[;k nh tkrh gS",  # Hindi header (fixed escaping)
        r"भारत का राजपत्र",
        r"असाधारण",
        r"भाग\s*II—खण्ड\s*1",
        r"प्राधिकरण द्वारा प्रकाशित",
        r"नई दिल्ली,\s*दिनाांक\s*\d{1,2}\s*[A-Za-z]+\s*\d{4}",
        r"पंजीकरण सं\.\s*डी.एल-\s*\d+",
    ]

    for i, start_info in enumerate(section_starts):
        section_start_index = start_info["index"]
        section_number = start_info["number"]

        # End index = start of next section, or EOF
        section_end_index = len(text_content)
        if i + 1 < len(section_starts):
            section_end_index = section_starts[i + 1]["index"]

        # If a chapter starts before the next section, cut earlier
        for chap in chapter_info:
            if chap["start"] > section_start_index and chap["start"] < section_end_index:
                section_end_index = chap["start"]
                break

        section_text_content = text_content[section_start_index:section_end_index].strip()

        # Remove unwanted headers/footers
        for pattern in unwanted_patterns:
            section_text_content = re.sub(pattern, "", section_text_content, flags=re.MULTILINE | re.DOTALL).strip()

        # Normalize spacing
        section_text_content = re.sub(r'\n\s*\n+', '\n\n', section_text_content)

        # Determine chapter
        current_chapter = "PRELIMINARY"
        for chap in chapter_info:
            if section_start_index >= chap["start"]:
                current_chapter = chap["title"]
            else:
                break

        sections.append({
            "section_id": f"Section {section_number}",
            "chapter": current_chapter,
            "text": section_text_content
        })

    return sections

def save_sections_to_json(sections, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_pdf = "ConsumerProtection Act 2019.pdf"
    output_json = "consumer_protection_act_sections.json"

    try:
        print("Extracting text from PDF...")
        cpa_text = extract_text(input_pdf)

        print("Parsing sections...")
        parsed_sections = parse_cpa_sections(cpa_text)

        save_sections_to_json(parsed_sections, output_json)
        print(f"✅ Successfully parsed {len(parsed_sections)} sections into '{output_json}'")

    except FileNotFoundError:
        print(f"❌ Error: File not found at '{input_pdf}'")
    except Exception as e:
        print(f"❌ An error occurred: {e}")