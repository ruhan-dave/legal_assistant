import streamlit as st
import cohere
from classic_rag import process_pdf_to_text, customize_chunking, retrieve_top_chunks, get_llm_output
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
import asyncio
from qdrant_client import QdrantClient

# Set up asyncio event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Access API Key from Streamlit Secrets
cohere_key = st.secrets["cohere"]["api_key"] 
qdrant_key = st.secrets["qdrant"]["key"]
qdrant_host = st.secrets["qdrant"]["host"]

ch = cohere.ClientV2(api_key=cohere_key)

client = QdrantClient(
    url=qdrant_host,
    api_key=qdrant_key,
    timeout=50
)

def extract_page_text(args):
    """Helper function for parallel PDF page extraction."""
    reader, page_num = args
    try:
        page = reader.pages[page_num]
        return " ".join(page.extract_text().split())  # Remove newlines and extra spaces
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")  # Error handling
        return ""
    
def main():
    st.title("Hi, I'm Your Virtual Legal Affairs Assistant")
    st.write("Upload a PDF file containing legality, compliance, and rental/lease agreement information, then ask me questions!")

    st.sidebar.header("Files in the Local Folder")
    
    # Upload PDF file
    pdffile = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdffile is not None:
        st.write("Processing...")
        try:
            pdf_reader = PdfReader(pdffile)
            num_pages = len(pdf_reader.pages)
            st.write(f"Extracted {num_pages} pages from the PDF file.")
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return  # Exit function early if PDF cannot be processed
        
        # Process PDF text
        with ThreadPoolExecutor(max_workers=4) as executor:
            pdftext = list(executor.map(extract_page_text, [(pdf_reader, i) for i in range(num_pages)]))
            pdftext = " ".join(pdftext)
        
        st.write(pdftext)

        list_chunks = customize_chunking(pdftext)

        st.write(f"Generated {len(list_chunks)} text chunks from the PDF.")
        if not list_chunks:
            st.error("No text chunks were generated from the PDF.")
            return
        
        # Accept user query
        user_query = st.text_input("Ask a question about the document:")
        if user_query:
            st.write("Thinking...")

            # Retrieve relevant chunks
            top_chunks = retrieve_top_chunks(
                client=client,
                query=user_query, 
                collection_name="new-collection", 
                list_chunks=list_chunks, 
                n=5
            )

            results = [chunk for chunk in top_chunks]
    
            st.write(f"Found {len(results)} relevant chunks: {results}")

            if not top_chunks:
                st.error("No relevant information found. Try rephrasing your query.")
                return

            # Get LLM output
            response = get_llm_output(top_chunks, ch, user_query)
            st.success("Here's what you need to know:")
            st.write(response)

if __name__ == "__main__":
    main()
