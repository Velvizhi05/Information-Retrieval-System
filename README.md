# 📘 Information Retrieval System

A **PDF-based Question Answering System** built with **LangChain**, **Streamlit**, and **vector databases**.  
Upload one or more PDF files, process them into embeddings, and ask natural language questions — the system retrieves the most relevant context and generates accurate answers.  

---

## ✨ Features
- 📄 **PDF Upload**: Supports multiple PDF files.  
- 🧠 **Text Chunking**: Splits large documents into manageable chunks.  
- 🔍 **Vector Search**: Uses embeddings for semantic similarity search.  
- 🤖 **Conversational QA**: Maintains chat history across turns.  
- ⚡ **Streamlit UI**: Clean and interactive interface.  

---

## 🛠️ Tech Stack
- [Python 3.10+](https://www.python.org/downloads/)  
- [Streamlit](https://streamlit.io/) – Web UI  
- [LangChain](https://www.langchain.com/) – LLM orchestration  
- [SambaNova API](https://sambanova.ai/) – LLM provider  
- [FAISS / Chroma](https://python.langchain.com/docs/integrations/vectorstores) – Vector database  

---

## 📂 Project Structure
Information-Retrieval-System/
│
├── app.py # Streamlit main app
|__research/
|  |__trails.ipynb
├── src/
│ ├── helper.py # Helper functions (LLM wrapper, chains, vectorstore)
│ └── init.py
├── requirements.txt # Python dependencies
└── README.md # Documentation


---

## ⚡ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Velvizhi05/Information-Retrieval-System.git
cd Information-Retrieval-System
```
### 2. Create and activate virtual environment
```
python -m venv genai_env
source genai_env/bin/activate      # Linux / Mac
genai_env\Scripts\activate         # Windows
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Add your API Key
```
Create a .env file in the root directory and add:
SAMBANOVA_API_KEY=your_api_key_here
```
### 5. Run the app
```
streamlit run app.py
```
🎯 Usage
```
1.Upload one or more PDF files from the sidebar.
2.Click Submit & Process to generate embeddings.
3.Ask natural language questions in the text input field.
4.View the interactive conversation history between you and the bot.
```
🖼️ Demo Screenshot
```
![App Screenshot]("[src/output.pn](https://github.com/Velvizhi05/Information-Retrieval-System/blob/main/src/output.png)g") 
```
📌 Roadmap
```
 Add live streaming answers in UI

 Support for multiple LLM providers

 Advanced PDF parsing (tables, images)
```
🤝 Contributing
```
Pull requests are welcome! For major changes, please open an issue first to discuss.
```
📜 License
```
This project is licensed under the MIT License.

Do you also want me to generate a **`requirements.txt`** (with exact versions of Streamlit, LangChain, FAISS, etc.) so anyone who clones your repo can run it without dependency conflicts?




