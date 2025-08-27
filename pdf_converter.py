import PyPDF2
import os

def convert_pdf_to_text(pdf_path, output_text_path):
    """
    Converts a PDF file to a plain text file.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_text_path (str): The path where the output text file will be saved.
    """
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() or "" # Added "" to handle None case

        with open(output_text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text_content)
        print(f"Successfully converted '{pdf_path}' to '{output_text_path}'")
    except FileNotFoundError:
        print(f"Error: PDF file not found at '{pdf_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- IMPORTANT: EDIT THESE PATHS ---
    # Make sure this is the correct path to your PDF file.
    # For example: 'C:\\Users\\ashir\\OneDrive\\Desktop\\legal-advisor\\ConsumerProtection Act 2019.pdf'
    # Or simply 'ConsumerProtection Act 2019.pdf' if it's in the same directory as this script.
    pdf_file_name = r'C:\Users\ashir\OneDrive\Desktop\legal-advisor_rework\ConsumerProtection Act 2019.pdf' # <--- CHANGE THIS TO YOUR PDF FILE NAME

    # This will be the name of the output text file.
    output_text_file_name = 'consumer_protection_act.txt' # <--- YOU CAN CHANGE THIS NAME

    # Construct full paths
    current_directory = os.path.dirname(os.path.abspath(__file__))
    pdf_input_path = os.path.join(current_directory, pdf_file_name)
    output_text_filepath = os.path.join(current_directory, output_text_file_name)

    convert_pdf_to_text(pdf_input_path, output_text_filepath)
