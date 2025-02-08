import streamlit as st
from backend.core_runnable import run_llm  # Import the function from your existing code

# 🌟 Streamlit UI Configuration
st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖", layout="wide")

# 🎨 UI Title and Description
st.title("💬 LangChain Chatbot with Streamlit")
st.write("🚀 Ask me anything! I retrieve documents and answer your questions using LangChain and Pinecone.")

# ✅ Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📝 User Input
user_input = st.text_input("Type your question:", key="user_input")

# 🎯 Handle Submit Button
if st.button("Ask"):
    if user_input:
        # Display user query
        st.session_state.chat_history.append(("🧑‍💻 You:", user_input))

        # Get response from the LLM
        response = run_llm(user_input)

        # Display bot response
        st.session_state.chat_history.append(("🤖 Bot:", response.content))

        # Clear input box
        st.rerun()

# 📜 Display Chat History
st.subheader("Chat History")
for role, message in st.session_state.chat_history:
    st.markdown(f"**{role}** {message}")
