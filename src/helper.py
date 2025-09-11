import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional, Mapping
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
SAMBA_API_KEY = os.getenv("SAMBA_API_KEY")

if not SAMBA_API_KEY:
    raise ValueError("❌ SAMBA_API_KEY not found. Please set it in your .env file.")

# Initialize SambaNova client
samba_client = OpenAI(
    api_key=SAMBA_API_KEY,
    base_url="https://api.sambanova.ai/v1",
)

# ---------- PDF Processing ----------
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split extracted text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """Create FAISS vector store using HuggingFace embeddings (local)."""
    if not text_chunks:
        raise ValueError("No text chunks were generated. Check if the PDF contains extractable text.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# ---------- SambaNova LLM Wrapper ----------
class SambaLLM(LLM):
    """LangChain-compatible wrapper for SambaNova OpenAI-compatible API."""
    client: Any
    model: str = "Llama-4-Maverick-17B-128E-Instruct"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "sambanova"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "temperature": self.temperature}


def get_conversation_chain(vectorstore):
    """Create a conversational retrieval chain with SambaNova LLM."""
    llm = SambaLLM(client=samba_client, model="Llama-4-Maverick-17B-128E-Instruct", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
