# Import the necessary Libraries
from utility.rag import RAG

# Provide pdf_folder_location
pdf_folder_location = "../_data/rag/cloud/"
persist_directory = "../_data/rag/db/reports_db"

if __name__ == "__main__":
    # Create the vectod embeddings
    rag = RAG(pdf_folder_location, persist_directory)
    rag.build_vector_db()  

    ##Now that the dB is build. let's make query and test it.
    query = "How is the company integrating AI across their various business units" 
    source_document = pdf_folder_location + "Meta-10-k-2023.pdf"

    docs = rag.test_query(query, source_document)

    # Print the retrieved docs, their source and the page number
    # (page number can be accessed using doc.metadata['page'] )
    for i, doc in enumerate(docs):
        print(f"Retrieved chunk {i+1}: \n")
        print(doc)
        print(doc.page_content.replace('\t', ' '))
        print("Source: ", doc.metadata['source'],"\n ")
        print("Page Number: ",doc.metadata['page'],"\n===================================================== \n")
        print('\n')