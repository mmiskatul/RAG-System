from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings 


persistent_directory ="db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",  
        model_kwargs={"device": "cpu"}  
    )

db =Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"} 
)

query ="which island does SpaceX lease for its launches in the Pacific?"

retriver =db.as_retriever(search_kwargs={"k":5})

# retriver =db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k":5,
#         "score_threshold":0.3
#     }
# )

relevant_docs  =retriver.invoke(query)

print("--- Context ---")

for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")