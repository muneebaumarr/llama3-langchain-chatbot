import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"

memory = ConversationBufferMemory()

llm = OllamaLLM(model="llama3")

#conversation chain

conversation = ConversationChain(
    llm = llm,
    memory = memory,
    verbose = True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "input_text" not in st.session_state:
    st.session_state.input_text = ""

#creating chatbot

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please Provide response to the user quiries."),
        ("user", "question:{question}")
    ]

)


#GUI using streamlit

st.title("AI Assistant🤖")
st.subheader("Powered by LLaMA 3 + LangChain")

#chat history
for i, msg in enumerate(st.session_state.chat_history):
    speaker = "🧑 You" if i % 2 == 0 else "🤖 Bot"
    st.markdown(f"**{speaker}:** {msg}")


input_text = st.text_input("Enter your query...", value="", key=str(len(st.session_state.chat_history)))


if input_text:
    response = conversation.predict(input = input_text)
    st.write(response)

    # Update chat history
    st.session_state.chat_history.append(input_text)
    st.session_state.chat_history.append(response)
    
    st.session_state.input_text = ""
    st.rerun()

with st.expander("Conversation History"):
    st.markdown(memory.buffer)


