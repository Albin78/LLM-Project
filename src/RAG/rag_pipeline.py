from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List
from sentence_transformers import SentenceTransformer


class ModelEmbeddings(Embeddings):
    """Embeddings wrapper for the vector store"""


    def __init__(self, model_name: str = "all-MiniLM-L6-v2") :
        super().__init__()
        
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embedd list of texts"""
        embedds = self.model.encode(texts)
        return embedds.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embedd a single query"""
        embeds = self.model.encode([text])
        return embeds[0].tolist()


# Document Processor
class DocumentProcessor:
    """Process the medical documents for retrieval"""

    def __init__(self, chunk_size: int = 400, 
                chunk_overlap: int = 40) :


        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=[
            "\n\n",
            "\n",
            ".",
            "!",
            "?",
            ":",
            " ",
            ""
            ]
        )

    def load_and_split_document(self, file_path: str) -> List[Document]:
        """Load and split document"""

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path=file_path)
        else:
            loader = TextLoader(file_path=file_path)

        documents = loader.load()
        split_documents = self.text_splitter.split_documents(documents=documents)

        for i, split in enumerate(split_documents):
            split.metadata.update({
                "chunk_id": i,
                "source_type": "medical data",
                "chunk_size": len(split.page_content)
            })

        return split_documents


def create_prompt_template():
    """Creating a prompt template that fits the model"""

    template =  """

You are a medical assistant. Use ONLY the provided context to answer the question.

Question: {question}

Context: {context}

Answer:

"""

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["question", "context"]
    )

    return prompt_template