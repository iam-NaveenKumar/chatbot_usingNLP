import nltk
import random
import os
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ✅ Fix SSL issue
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath('nltk_data'))
nltk.download('punkt')

# ✅ Initialize Chatbot Model
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# ✅ Define chatbot intents (Expanded & Fixed)
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "What's up?", "How are you?", "Howdy"],
        "responses": ["Hi there!", "Hello!", "Hey!", "Not much, you?", "I'm doing well, thanks!"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you", "Goodbye", "Take care", "Catch you later"],
        "responses": ["Goodbye!", "See you soon!", "Take care!", "Have a great day!"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "I appreciate it", "Thanks a lot"],
        "responses": ["You're welcome!", "No problem!", "Glad to help!", "Anytime!"]
    },
    {
        "tag": "small_talk",
        "patterns": ["How was your day?", "What are you doing?", "Are you busy?"],
        "responses": ["My day has been great, thanks for asking!", "Just chatting with you!", "I'm never too busy to chat!"]
    },
    {
        "tag": "food",
        "patterns": ["What's your favorite food?", "Do you like Food?", "What's the best dish?"],
        "responses": ["I don't eat, but I've heard indian foods are amazing!", "People say Briyani is great!", "Food is the way to happiness, right?"]
    },
    {
        "tag": "sports",
        "patterns": ["Do you like football?", "Who is the best player?", "What's your favorite sport?"],
        "responses": ["I can't play, but football seems exciting!", "Messi,Ronaldo and Neymar are legends!", "Sports are a great way to stay active!"]
    },
    {
        "tag": "motivation",
        "patterns": ["I'm feeling down", "I need motivation", "Can you inspire me?"],
        "responses": ["You're stronger than you think!", "Tough times don’t last, but tough people do!", "Believe in yourself!"]
    },
    {
        "tag": "funny",
        "patterns": ["Tell me something funny", "Make me laugh", "Say something sarcastic"],
        "responses": ["Why don’t scientists trust atoms? Because they make up everything!",
                      "If laziness was a subject, I’d top the class!",
                      "I tried to be normal once. Worst two minutes of my life!"]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like?", "Is it raining?", "How's the temperature?"],
        "responses": ["I can't check real-time weather, but you can use a weather app!", "Try asking Siri or Google Assistant for live updates!"]
    },
    {
        "tag": "hobbies",
        "patterns": ["What are your hobbies?", "Do you have a hobby?", "What do you like to do?"],
        "responses": ["I love chatting with people like you!", "Helping users is my favorite thing to do.", "I don't have hobbies, but I can talk about yours!"]
    },
    {
        "tag": "chatbot_emotion",
        "patterns": ["Are you happy?", "Are you sad?", "Do you feel emotions?"],
        "responses": ["I don't have emotions, but I can try to make your day better!",
                      "I'm just an AI, but I always try to stay positive!",
                      "I don’t feel emotions, but I can understand them!"]
    },
    {
        "tag": "happiness",
        "patterns": ["I'm happy!", "I feel great", "Today is awesome"],
        "responses": ["That's amazing!", "Keep spreading positivity!", "Happiness is contagious!"]
    },
    {
        "tag": "sadness",
        "patterns": ["I'm feeling down", "I'm sad", "I need cheering up"],
        "responses": ["It's okay, bad days pass!", "Stay strong, you're doing great!", "I'm here for you!"]
    },
    {
        "tag": "loneliness",
        "patterns": ["I'm lonely", "I have no friends", "I feel alone"],
        "responses": ["You're not alone! I'm here to chat!", "Remember, people care about you!", "Let's talk! What's on your mind?"]
    },
]

# ✅ Train Model
tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)


# ✅ Chatbot function
def chatbot(input_text):
    input_vector = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_vector)[0]

    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])

    return "Sorry, I don't understand that."


# ✅ Streamlit Web UI
st.title("🤖 AI Chatbot")
st.write("💬 Ask me anything!")

# ✅ Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ✅ User input & Send button
user_input = st.text_input("Type your message:", key="input")

if st.button("Send"):
    if user_input:
        # ✅ Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # ✅ Generate chatbot response
        response = chatbot(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

# ✅ Add Clear Chat Button
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.rerun()
