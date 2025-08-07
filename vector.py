from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

def load_and_prepare(file_path, source_label):
    df = pd.read_csv(file_path)
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content = f"{row['company']} {row['sector']} {row['closing_stock_price']} {row['popularity_score']}",
            metadata={
                "closing stock price": row["closing_stock_price"],
                "sector": row["sector"],
                "source": source_label  # distinguish between stocks and stocks2
            },
            id=f"{source_label}_{i}"
        )
        documents.append(document)
        ids.append(f"{source_label}_{i}")
    
    return documents, ids

# Load and prepare both datasets
docs1, ids1 = load_and_prepare("stocks.csv", "stocks")
docs2, ids2 = load_and_prepare("stocks2.csv", "stocks2")

vector_store = Chroma(
    collection_name="stocks",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents only on first run
if add_documents:
    vector_store.add_documents(documents=docs1 + docs2, ids=ids1 + ids2)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
