from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from datetime import datetime
from src.RAG.llm import GPTModelLLM
from src.RAG.rag_pipeline import create_prompt_template, DocumentProcessor, ModelEmbeddings
import logging

logging.basicConfig(
    level=logging.INFO,
    format="- %(levelname)s - %(message)s"
)

class RAGPipeline:
    """The pipeline in which whole RAG works"""

    def __init__(self, vector_store_path: Optional[str],
                max_length: int=1024,
                temperature: float=None,
                top_p: float=None,
                top_k: int=None):
        
        self.llm = GPTModelLLM(max_length=max_length,
                               temperature=temperature,
                               top_k=50, top_p=0.9,
                               max_new_tokens=300)
        
        self.embeddings = ModelEmbeddings()
        self.processor = DocumentProcessor()
        self.vector_store = None
        self.qa_chain = None

        if vector_store_path:
            self.load_vector_store(path=r"src\RAG")

    def build_knowledge(self, document_path: str,
                        save_path: str):
        
        """Building vectore store"""

        logging.info("Loading and processing documents")

        documents = self.processor.load_and_split_document(document_path)
        logging.info(f"Created {len(documents)} chunks")

        self.vector_store = FAISS.from_documents(
            documents=documents, embedding=self.embeddings
        )

        if save_path:
            self.vector_store.save_local(save_path)
            logging.info(f"Successfully saved to {save_path}")

        self.create_qa_chain()
        logging.info("Successfully Build the knowledge base")

    def load_vector_store(self, path: str):
        """Loading the vector store from saved path"""

        self.vector_store = FAISS.load_local(path, self.embeddings,
                                            allow_dangerous_deserialization=True)
        self.create_qa_chain()

    def create_qa_chain(self):
        """QA chain with all wrapped up"""

        prompt = create_prompt_template()

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    
    def query_format(self, question: str):
        """Passes query and fetches the response"""

        if not self.qa_chain:
            return {"error": "knowledge base not implemented properly"}
        
        try:
            start_time = datetime.now()

            result = self.qa_chain.invoke({"query": question})
            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "question": question,
                "answer": result["result"],
                "processing_time": processing_time,
                "sources": [
                    {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ],
                "num_sources": len(result["source_documents"]),
                "success": True
            }
        
        except Exception as e:
            return {
                "question": question,
                "result": "Error while processing question",
                "error": str(e),
                "success": False
            }