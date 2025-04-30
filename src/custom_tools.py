# This file is currently empty as we're using smolagents' built-in tools
# We'll implement custom tools here later when needed
from smolagents import Tool
import os
from pathlib import Path
from typing import Dict, List, Optional
from docling.document_converter import DocumentConverter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob


class CalculatorTool(Tool):

    name = "calculator"
    description = """This is a calculator tool. Use it to perform basic arithmetic operations. 
        You can ask it to add, subtract, multiply, or divide numbers. It returns the numeric output of the operation."""

    inputs = {
        "num1": {
            "type": "number",
            "description": "The first number to be used in the operation.",
        },
        "num2": {
            "type": "number",
            "description": "The second number to be used in the operation.",
        },
        "operation": {
            "type": "string",
            "description": "The operation to perform. Can be 'add', 'sub', 'mul', or 'div'.",
        },
    }
    output_type = "number"

    def forward(self, num1: float, num2: float, operation: str) -> float:
        import operator

        operator_dict = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.truediv,
        }

        if operation not in operator_dict:
            raise ValueError(
                f"Invalid operation: {operation}. Must be one of {list(operator_dict.keys())}."
            )

        return operator_dict[operation](num1, num2)


class DocumentRetrieverTool(Tool):
    """
    A tool for retrieving relevant context from PDF documents without using an LLM.
    """

    name = "document_retriever"
    description = """This tool processes PDF documents and retrieves relevant context based on a query.
        It does not use an LLM to generate answers but returns the most relevant text passages from the documents."""

    inputs = {
        "query": {
            "type": "string",
            "description": "The query to find relevant information in the documents.",
        },
        "pdf_path": {
            "type": "string",
            "description": "Path to a PDF file or directory containing PDFs. If a directory, all PDFs in it will be processed.",
            "required": False,
            "nullable": True,
        },
        "num_results": {
            "type": "integer",
            "description": "Number of most relevant text chunks to return.",
            "required": False,
            "nullable": True,
        },
        "chunk_size": {
            "type": "integer",
            "description": "Size of text chunks for splitting documents.",
            "required": False,
            "nullable": True,
        },
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.converter = DocumentConverter()
        self.vector_store = None

    def process_pdf(self, pdf_path: str, max_pages: int = 100) -> str:
        """Process a single PDF document and return its text content."""
        try:
            result = self.converter.convert(pdf_path, max_num_pages=max_pages)
            return result.document.export_to_markdown()
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return ""

    def process_pdf_directory(
        self, directory_path: str, pdf_pattern: str = "*.pdf"
    ) -> Dict[str, str]:
        """Process all PDF files in a directory."""
        pdf_files = glob.glob(os.path.join(directory_path, pdf_pattern))
        documents = {}

        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            print(f"Processing {filename}...")
            text = self.process_pdf(pdf_file)
            if text:
                documents[filename] = text

        return documents

    def create_vector_store(self, documents: Dict[str, str]):
        """Create a vector store from the processed documents."""
        texts = []
        metadatas = []

        for filename, content in documents.items():
            chunks = self.text_splitter.split_text(content)
            texts.extend(chunks)
            metadatas.extend([{"source": filename}] * len(chunks))

        if texts:
            self.vector_store = FAISS.from_texts(
                texts, self.embedding_model, metadatas=metadatas
            )
            print(f"Vector store created with {len(texts)} text chunks")
            return True
        else:
            print("No text chunks to create vector store")
            return False

    def forward(
        self,
        query: str,
        pdf_path: Optional[str] = None,
        num_results: int = 3,
        chunk_size: int = 1000,
    ) -> List[Dict]:
        """
        Retrieve relevant context from documents based on a query.

        Args:
            query: The query string for searching relevant information
            pdf_path: Path to a PDF file or directory containing PDFs
            num_results: Number of most relevant chunks to return
            chunk_size: Size of text chunks for document splitting

        Returns:
            List of dictionaries containing relevant text chunks and metadata
        """
        # Update chunk size if provided
        if chunk_size != 1000:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=min(200, chunk_size // 5),
                length_function=len,
            )

        # Process documents if pdf_path is provided
        if pdf_path:
            if os.path.isdir(pdf_path):
                documents = self.process_pdf_directory(pdf_path)
            elif os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
                documents = {os.path.basename(pdf_path): self.process_pdf(pdf_path)}
            else:
                return [
                    {
                        "error": f"Invalid path: {pdf_path}. Must be a PDF file or directory."
                    }
                ]

            # Create vector store
            success = self.create_vector_store(documents)
            if not success:
                return [
                    {
                        "error": "Failed to create vector store. No text chunks extracted."
                    }
                ]

        # Ensure vector store exists
        if not self.vector_store:
            # Try to find documents in the default directory
            documents_dir = Path("./documents")
            if documents_dir.exists() and documents_dir.is_dir():
                documents = self.process_pdf_directory(str(documents_dir))
                success = self.create_vector_store(documents)
                if not success:
                    return [{"error": "No documents found in default directory."}]
            else:
                return [{"error": "No vector store created and no PDF path provided."}]

        # Retrieve relevant documents
        results = self.vector_store.similarity_search_with_score(query, k=num_results)

        # Format the results
        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "similarity_score": score,
                }
            )

        return formatted_results
