from transformers import pipeline

# Initialize the summarization pipeline globally to avoid reloading the model
# We'll use a smaller model suitable for local execution if a GPU is not available
# For better quality, consider larger models or cloud-based LLMs
summarizer = None

def initialize_summarizer():
    global summarizer
    if summarizer is None:
        try:
            # Using a smaller, faster model for summarization
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", tokenizer="sshleifer/distilbart-cnn-12-6")
            print("✅ LLM Summarizer initialized successfully.")
        except Exception as e:
            print(f"❌ Error initializing LLM summarizer: {e}")
            print("Please ensure you have an internet connection for the first run to download the model.")
            summarizer = None
    return summarizer

def generate_layman_summary(text, query):
    """
    Generates a plain-language summary of the given text using a pre-trained LLM.
    The query can be used to guide the summarization if the model supports it.
    """
    global summarizer
    if summarizer is None:
        summarizer = initialize_summarizer()
        if summarizer is None:
            return "(Unable to generate summary - LLM not initialized)"

    # Combine text and query if the model can use query as context, otherwise just summarize text
    # For distilbart, we primarily summarize the text directly.
    # You might need to experiment with prompt engineering for query-aware summarization.
    input_text = f"Summarize the following legal text in simple, layman's terms, focusing on how it relates to: {query}. Text: {text}"
    
    # Limiting input length for the LLM as models have context window limits
    # Distilbart's max input length is 1024 tokens. Adjust as needed.
    max_input_length = 512 # A safe limit for many smaller models
    if len(input_text.split()) > max_input_length:
        # Simple truncation for demonstration. For production, consider more sophisticated chunking.
        input_text = ' '.join(input_text.split()[:max_input_length])

    try:
        # Generate summary. min_length and max_length can be adjusted.
        summary = summarizer(input_text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text'].strip()
    except Exception as e:
        print(f"❌ Error generating summary with LLM: {e}")
        return "(Error generating summary)"

if __name__ == "__main__":
    # Example usage
    initialize_summarizer()
    sample_legal_text = (
        "Section 85. A product service provider shall be liable in a product liability action, if— "
        "(a) the service provided by him was faulty or imperfect or deficient or inadequate "
        "in quality, nature or manner of performance which is required to be provided by or "
        "under any law for the time being in force, or pursuant to any contract or otherwise; or "
        "(b) there was an act of omission or commission or negligence or conscious "
        "withholding any information which caused harm; or "
        "(c) the service provider did not issue adequate instructions or warnings to "
        "prevent any harm; or "
        "(d) the service did not conform to express warranty or the terms and conditions "
        "of the contract."
    )
    sample_query = "My phone battery dies quickly after replacement."
    summary = generate_layman_summary(sample_legal_text, sample_query)
    print(f"\nGenerated Summary:\n{summary}")

    sample_legal_text_2 = (
        "Section 2. In this Act, unless the context otherwise requires,— (1) \"advertisement\" means any audio or visual publicity, "
        "representation, endorsement or pronouncement made by means of light, sound, smoke, gas, print, "
        "electronic media, internet or website and includes any notice, circular, label, wrapper, "
        "invoice or such other documents; (2) \"appropriate laboratory\" means a laboratory or an organisation— "
        "(i) recognised by the Central Government; or (ii) recognised by a State Government, subject to such guidelines as may "
        "be issued by the Central Government in this behalf; or (iii) established by or under any law for the time being in force, "
        "which is maintained, financed or aided by the Central Government or a State Government "
        "for carrying out analysis or test of any goods with a view to determining whether "
        "such goods suffer from any defect;"
    )
    sample_query_2 = "What is an advertisement under this act?"
    summary_2 = generate_layman_summary(sample_legal_text_2, sample_query_2)
    print(f"\nGenerated Summary 2:\n{summary_2}")
