from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings 
from transformers import pipeline

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

query ="In what year did Tesla begin production of the roadster?"

retriver =db.as_retriever(search_kwargs={"k":5})

# retriver =db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k":5,
#         "score_threshold":0.3
#     }
# )

relevant_docs  =retriver.invoke(query)
print(f"User Query : {query}")
print("--- Context ---")

for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    
    
## COMBINE the query and relavant document contents 
combine_input = f"""Base on the following documents ,please answer this question {query}

Documents: 
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please Provide a clear , helpful answer using only the information from these documents If you can't find the answer in the documents , say "I don't have enough information to answer the question based on the provided documents 
"""



# -----------------------------
# 3️⃣ Generate Answer
# -----------------------------
def generate_answer(prompt):
    """Use a local LLM to generate an answer"""
    generator = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")      
    response = generator(prompt, return_full_text=False)
    return response[0]["generated_text"]

print("\n=== Generating Answer ===")
answer = generate_answer(combine_input)

print("\n=== Final Answer ===")
print(answer)