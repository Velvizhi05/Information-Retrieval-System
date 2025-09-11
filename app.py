import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain


def user_input(user_question):
    # Run query
    response = st.session_state.conversation_chain.invoke({"question": user_question})
    
    # Save updated history from memory
    st.session_state.chat_history = st.session_state.conversation_chain.memory.chat_memory.messages

    # Display conversation
    for i, message in enumerate(st.session_state.chat_history):
        if message.type == "human":
            st.write(f"**User:** {message.content}")
        else:
            st.write(f"**Bot:** {message.content}")


def main():
    st.set_page_config("Information Retrieval System")
    st.header("Information Retrieval System 🤞💁‍♂️")

    # Session state setup
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User question input
    user_question = st.text_input("Ask your question about the document")
    if user_question and st.session_state.conversation_chain:
        user_input(user_question)

    # Sidebar - file upload and processing
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation_chain = get_conversation_chain(vectorstore)
                    st.success("Done")
                    st.snow()
            else:
                st.warning("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
