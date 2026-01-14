import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma 
from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("BAAI/bge-m3")

# sentences = [
#     "That is a happy person",
#     "That is a happy dog",
#     "That is a very happy person",
#     "Today is a sunny day"
# ]
# embeddings = model.encode(sentences)

# similarities = model.similarity(embeddings, embeddings)
# print(similarities.shape)
# # [4, 4]

def load_documents(docs_path="docs"):
    """Load the text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists 
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory with UTF-8 encoding
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}  # <-- fix applied here
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    # Preview first 2 documents
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" Metadata: {doc.metadata}")
    
    return documents

def split_documents(documents,chunk_size=800,chunk_overlap=0):
    """Split documents into smaller chunck with overlap"""
    print("Splitting documents into chunk")
    
    text_splitter =CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks =text_splitter.split_documents(documents)
    
    if chunks:
        
        for i,chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1}")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Context:")
            print(chunk.page_content)
            print("-"*50)
        if len(chunks)>5 : 
            print(f"\n... and {len(chunks)-5} more chunks")
    return chunks
 
def main():
    print("main function")
    
    #load the file 
    documents =load_documents(docs_path="docs")
    
    #chucking the file 
    chucks =split_documents(documents)

if __name__ == "__main__" :
    main()