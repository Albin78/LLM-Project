import streamlit as st
import json
import requests
import httpx
import asyncio
import os
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MED CHAT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.welcome-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 40vh;
    text-align: center;
    color: gray;
    font-size: 3.5rem;
    font-weight: 500;
    }
.main-header {
    font-size: 3.0rem;
    font-weight: bold;
    margin-bottom: 1rem;
    text-align: center;
    color: #1f77b4;
    }
    .chat-container {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    background-color: #f8f9fa;
    }
    .user-message {
    background-color: #e3f2fd;
    color: #000;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.5rem 0;
    }
    .assistant-message {
    background-color: #f3e5f5;
    color: #000;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.5rem 0;
    }
</style> 
""", unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_connected" not in st.session_state:
    st.session_state.api_connected = False

if "welcome_message" not in st.session_state:
    st.session_state.welcome_message = True

def check_api_health() -> bool:
    """Check Api Health status"""
    
    try:
        response = requests.get(url=f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data =  response.json()
            return data.get("pipeline", False)
    
    except requests.exceptions.RequestException:
        return False

def query_to_model(query: str, config: Dict[str, Any]):
    """Query to process"""

    try:
        payload = {
            "query": query,
            "max_new_tokens": config["max_new_tokens"],
            "temperature": config["temperature"],
            "top_k": config["top_k"],
            "top_p": config["top_p"]
        }

        response = requests.post(
            url=f"{API_BASE_URL}/query",
            json=payload, timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}, response text- {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

async def query_stream(query: str, config: Dict[str, Any]):
    """Processing query in stream effect"""
    
    try:

        placeholder = st.empty()
        text_response = ""
        payload = {
            "query": query,
            "max_new_tokens": config["max_new_tokens"],
            "temperature": config["temperature"],
            "top_k": config["top_k"],
            "top_p": config["top_p"]
        }

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(method="POST", url=f"{API_BASE_URL}/stream",
                                    json=payload) as resposnse:
                async for line in resposnse.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "chunk" in data:
                            text_response += data["chunk"]
                            placeholder.markdown(f'<div class="assistant-message">{text_response}</div>',
                                                unsafe_allow_html=True)
                        
                        if "time" in data:
                            process_time = data["time"]
                        
                        if data.get("done"):
                            break
        
        return text_response, process_time

    except httpx._exceptions.HTTPError as e:
        st.error(f"Error occurred during quering process stream: {e}")
        return None

def main():
    """Main function which holds whole running pipeline"""

    st.markdown('<div class="main-header">🤖 MED CHAT</div>', unsafe_allow_html=True)
    if st.session_state.welcome_message and not st.session_state.messages:
        st.markdown('<div class="welcome-container">Where Should we begin</div>',
                    unsafe_allow_html=True)

    with st.sidebar:

        st.subheader("🔗 API Status")
        if st.button("Check Connection"):
            with st.spinner("Checking Connection ⏳"):
                st.session_state.api_connected = check_api_health()

        status_color = "🟢" if st.session_state.api_connected else "🔴"
        status_text = "Connected" if st.session_state.api_connected else "Disconnected"
        st.write(f"{status_color} **{status_text}**")

        if not st.session_state.api_connected:
            st.warning(f"Api connection is failed, check your connection {API_BASE_URL}")

        st.divider()

        with st.expander("⚙️ Settings"):
            
            st.subheader("🛠️ Model parameters")
            max_new_tokens = st.slider("Max Tokens", min_value=50, max_value=400, value=300, step=50)
            temperature = st.slider("Temperature", min_value=0.2, max_value=2.0, value=0.7, step=0.1)
            top_p = st.slider("Top-P", min_value=0.2, max_value=2.0, value=0.9, step=0.1)
            top_k = st.slider("Top-K", min_value=10, max_value=50, value=30, step=5)

        settings = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

        st.divider()

        if st.button("🗑️ New Chat"):
            st.session_state.messages = []
            st.session_state.welcome_message = True
            st.rerun()
    
    st.subheader("Chat History")
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>🧑:</strong> {message["content"]}</div>', 
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>🕵🏼‍♂️:</strong> {message["content"]}</div>',
                            unsafe_allow_html=True)

    st.subheader("✍️ Ask a question")
    user_query = st.chat_input(
        "Enter you question"
    )
    
    if user_query:

        if not st.session_state.api_connected:
            st.error("API connection failed, Check your api connection")

        else:
            st.session_state.messages.append({
                "role": "user",
                "content": user_query,
                "timestamp": datetime.now().isoformat()
            })

            st.session_state.welcome_message = False

            with st.spinner("🧠 Processing query"):
                result, process_time = asyncio.run(query_stream(query=user_query,
                                        config=settings))

                if result:
                    assistant_message = {
                        "role": "assistant",
                        "content": result,
                        "timestamp": datetime.now().isoformat()
                    }

                    st.session_state.messages.append(assistant_message)
                    st.success(f"Response delivered in {process_time:.2f} s") 
                
                else:
                    st.error("Failed to get the response")
    
            st.rerun()

if __name__ == "__main__":
    main()
                    

    


                    



