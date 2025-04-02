import sys
import os

# Force the use of pysqlite3 instead of system sqlite3
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    print("pysqlite3 not found. Ensure it is installed.")



import streamlit as st
import openai
import os
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db1")  # Ensure ChromaDB is set up
collection = client.get_collection("sentiment_analysis")

# Custom CSS for WhatsApp-style chat UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
    body { font-family: 'Roboto', sans-serif; }
    .chat-container { max-width: 600px; margin: auto; }
    .user-msg { background-color: #DCF8C6; text-align: right; padding: 12px; border-radius: 12px; margin: 5px 0; float: right; clear: both; max-width: 70%; }
    .bot-msg { background-color: #E5E5EA; text-align: left; padding: 12px; border-radius: 12px; margin: 5px 0; float: left; clear: both; max-width: 70%; }
    .sentiment-box { display: block; padding: 6px; border-radius: 6px; font-weight: bold; margin: 10px auto; max-width: 200px; text-align: center; }
    .positive { background-color: #D4EDDA; color: #155724; } 
    .neutral { background-color: #FFF3CD; color: #856404; }
    .negative { background-color: #F8D7DA; color: #721C24; }
    .clear { clear: both; }
    </style>
""", unsafe_allow_html=True)

st.title("💬 WhatsApp-Style Sentiment Analysis Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
    st.markdown(f'<div class="{css_class}">{msg["content"]}</div><div class="clear"></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="user-msg">{user_input}</div><div class="clear"></div>', unsafe_allow_html=True)

    # Query ChromaDB for similar reviews
    query_embedding = openai.embeddings.create(
        input=user_input, model="text-embedding-3-small"
    ).data[0].embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    # **Fix: Extract from `metadatas` instead of `documents`**
    similar_reviews = []
    if results and "metadatas" in results and results["metadatas"]:  # Ensure results exist
        for meta in results["metadatas"][0]:  # Iterate over metadata list
            if meta and "review" in meta and "sentiment" in meta:  # Ensure fields exist
                similar_reviews.append(f'- "{meta["review"]}" (Sentiment: {meta["sentiment"]})')
    else:
        similar_reviews.append("No similar reviews found.")

    similar_texts = "\n".join(similar_reviews)


    # **Sentiment Analysis using OpenAI**
    sentiment_prompt = f"""
    Analyze the sentiment of the following text: "{user_input}"
    
    Here are similar past reviews and their sentiments:
    {similar_texts}
    
    Predict the sentiment as Positive, Negative, or Neutral.
    """
    
    sentiment_response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "Classify sentiment as Positive, Negative, or Neutral. and just return sentiment"},
                  {"role": "user", "content": sentiment_prompt}]
    )
    sentiment_text = sentiment_response.choices[0].message.content.strip()

    # **Sentiment Mapping**
    sentiment_emojis = {"Positive": "🟢😊", "Neutral": "🟡😐", "Negative": "🔴😠"}
    sentiment_classes = {"Positive": "positive", "Neutral": "neutral", "Negative": "negative"}
    sentiment_emoji = sentiment_emojis.get(sentiment_text, "⚪🤖")
    sentiment_class = sentiment_classes.get(sentiment_text, "neutral")

    # **Generate AI response**
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=st.session_state.messages
    )
    bot_reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # **Display sentiment and AI response**
    st.markdown(f'<div class="sentiment-box {sentiment_class}">{sentiment_emoji} {sentiment_text}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-msg">{bot_reply}</div><div class="clear"></div>', unsafe_allow_html=True)
