import streamlit as st
import cohere
from classic_rag import process_pdf_to_text, customize_chunking, retrieve_top_chunks, get_llm_output

# Access API Key from Streamlit Secrets
cohere_key = st.secrets["cohere_key"]  # Retrieves securely stored key

def main():
    ch = cohere.Client(cohere_key)

    st.title("Hi, I'm Your Virtual Legal Affairs Assistant")
    st.write("Upload a PDF file containing legality, compliance, and rental/lease agreement information, then ask me questions!")

    st.sidebar.header("Files in the Local Folder")
    pdffiles = st.file_uploader("Upload your PDF(s) here:", type="pdf", accept_multiple_files=True)
    user_query = st.text_input("Enter your question here:", value="what are my lease terms?")

    if pdffiles:
        # Process multiple PDFs and combine their text
        pdftext = "\n".join([process_pdf_to_text(pdf) for pdf in pdffiles])
        list_chunks = customize_chunking(pdftext)

        if st.button("Ask"):
            try:
                with st.spinner("Thinking..."):
                    top_chunks = retrieve_top_chunks(
                        query=user_query, 
                        collection_name="daves-rag", 
                        chunks=list_chunks, 
                        n=5
                    )
                    response = get_llm_output(top_chunks, ch, user_query)

                st.success("Here's what you need to know:")
                st.write(response)
            except Exception as e:
                st.error(f"Oops! Something went wrong. Please try again: {e}")

if __name__ == "__main__":
    main()
