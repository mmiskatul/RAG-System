from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings 


persistent_directory ="db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",  
        model_kwargs={"device": "cpu"}  
    )