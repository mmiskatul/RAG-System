import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma 
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

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

def main():
    print("main function")

if __name__ == "__main__" :
    main()