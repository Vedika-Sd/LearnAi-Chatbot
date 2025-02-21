# Import necessary libraries
import os  
import json  
import datetime
import csv 
import nltk 
import ssl 
import streamlit as st  
import random  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  
from nltk.tokenize import word_tokenize  

# Download necessary NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load intents from JSON file
# File contains predefined patterns and responses for the chatbot
file_path = os.path.abspath("intents.json")
with open(file_path, 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Preprocessing function
def preprocess(text):
    """
    Preprocesses input text by:
    - Tokenizing into words
    - Converting to lowercase
    - Removing stopwords and non-alphabetic tokens
    - Lemmatizing words
    """
    tokens = word_tokenize(text.lower()) 
    stop_words = set(stopwords.words('english'))  
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    lemmatizer = WordNetLemmatizer() 
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens] 
    return ' '.join(lemmatized_tokens)

# Extract patterns and tags from intents
# Patterns are training examples, and tags are their corresponding labels
patterns, tags = [], []
for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(preprocess(pattern))  # Preprocess each pattern
        tags.append(intent["tag"])  # Associate each pattern with a tag

# Vectorize and train the model
vectorizer = TfidfVectorizer()  
x = vectorizer.fit_transform(patterns)  
model = LogisticRegression(random_state=0, max_iter=1000)  
model.fit(x, tags) 

# Chatbot function to handle user input
def chatbot(input_text):
    """
    Processes user input, predicts the intent, and retrieves an appropriate response.
    """
    input_text = preprocess(input_text) 
    input_vector = vectorizer.transform([input_text]) 
    tag = model.predict(input_vector)[0]  
    for intent in intents:
        if intent['tag'] == tag:  
            response = random.choice(intent['responses'])  
            return response.replace('\n', '<br>')  
    return "Sorry, I didn't understand that. Could you rephrase?"  # Default response for unknown inputs

# Streamlit app
def main():
    """
    Streamlit-based user interface for the chatbot.
    """
    st.title("LearnAI: Your AI/ML Study Assistant") 
    st.sidebar.title("Menu")  
    menu = ["Home", "Conversation History", "About"]  
    choice = st.sidebar.selectbox("Menu", menu)  

    if choice == "Home":
        st.subheader("Start Chatting")  
        
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        # User input field
        user_input = st.text_input("You:", key="user_input")
        if user_input:
            response = chatbot(user_input)  
            st.markdown(f"**Chatbot:** {response}", unsafe_allow_html=True)  

            # Log the conversation with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

    elif choice == "Conversation History":
        st.subheader("Conversation History") 
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  
                rows = list(csv_reader) 
                reversed_rows = rows[::-1] # Reverse the list to show latest conversations first

                for row in reversed_rows:
                    st.text(f"User: {row[0]}")  
                    st.text(f"Chatbot: {row[1]}")  
                    st.text(f"Timestamp: {row[2]}") 
                    st.markdown("---") 

        except FileNotFoundError:
            st.write("No conversation history found.")  

    elif choice == "About":
        st.write("""
          ## About
         **LearnAI Assistance Chatbot** is an user-friendly virtual assistant designed to guide users in their journey of learning Artificial Intelligence (AI) and Machine Learning (ML). Powered by Natural Language Processing (NLP) techniques and a robust machine learning backend, it ensures accurate and engaging interactions to make AI/ML concepts accessible to everyone.

         ### How It Works
         1. **Data Loading**: The chatbot reads intents and patterns from a structured JSON file. These intents define the expected user queries and corresponding responses.
         2. **Preprocessing**: User input and training data are cleaned and normalized using tokenization, stopword removal, and lemmatization. This ensures consistent and meaningful text for processing.
         3. **Model Training**: A TF-IDF Vectorizer converts text into numerical features, enabling the chatbot to interpret and analyze the data. A Logistic Regression model is trained using hyperparameter optimization for accurate intent classification.
         4. **Chatting**: When a user enters a query, the chatbot preprocesses the input, predicts the intent, and retrieves a relevant response from its knowledge base. Randomized responses for each intent keep the conversation lively and engaging.

         ### Features
         - **Intent Recognition**: The chatbot accurately detects the intent behind user queries, ensuring helpful and precise responses.
         - **AI/ML Guidance**: It provides roadmaps, resources, and explanations for key AI/ML concepts.
         - **Conversation History**: All interactions are logged in a CSV file, allowing users to revisit and review past conversations. For user convenience, the latest interactions are displayed at the top.
         - **Streamlit Interface**: With a clean and intuitive web interface, users can interact with the chatbot seamlessly.

         ### Why LearnAI Assistance Chatbot?
         This chatbot is simple question-answering. It acts as a mentor for users new to the field of AI/ML, guiding them through:
         - Understanding fundamental concepts like supervised learning, unsupervised learning, deep learning, nlp, etc.
         - Suggesting learning resources such as courses, tutorials, and books.
         - Beginner-friendly with easy-to-understand responses.

         ### Future Improvements
         - **Advanced NLP Models**: Enhancing intent recognition and conversational abilities using models like BERT or GPT-based architectures.
         - **Context Awareness**: Enabling the chatbot to maintain the context of multi-turn conversations, improving user experience and relevance.
         - **Dynamic Recommendations**: Offering tailored learning paths based on user interactions and progress.
         - **User Authentication**: Allowing personalized user accounts for tracking individual learning history and preferences.
         - **Gamified Learning**: Introducing quizzes, badges, and leaderboards to make learning interactive and rewarding.

         ### Vision
         **LearnAI Assistance Chatbot** aspires to become a comprehensive AI/ML learning companion, democratizing knowledge and fostering curiosity in one of the most exciting fields of technology. By empowering learners with the right tools and guidance, it bridges the gap between **curiosity and expertise**.

         ### Mission
         To make AI and ML learning accessible, engaging, and effective for learners of all backgrounds, paving the way for innovation and growth in technology.

         ### Get Started
         Interact with LearnAI Assistance Chatbot today and take your first step toward mastering Artificial Intelligence and Machine Learning!
         """)
        
if __name__ == "__main__":
    main() 
