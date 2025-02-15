# Import the necessary Libraries
from utility.rag import RAG

# Provide pdf_folder_location
pdf_folder_location = "../_data/rag/cloud/"
persist_directory = "../_data/rag/db/reports_db"

if __name__ == "__main__":
    # Create the vectod embeddings
    rag = RAG(pdf_folder_location, persist_directory)
    
    user_question = "How is the company integrating AI across their various business units?"
    source_document = pdf_folder_location + "google-10-k-2023.pdf"

    answer = rag.question_and_answer(user_question, source_document)

    print(answer)

