# LearnAI - Your AI/ML Study Assistant

LearnAI is an interactive chatbot designed to assist students in learning Artificial Intelligence (AI) and Machine Learning (ML). Developed using **NLTK, TF-IDF, and Logistic Regression**, it provides guidance on AI/ML topics in an engaging manner. The chatbot is built with **Streamlit** for an intuitive user interface.

## Features
- **Intent-Based Responses**: Uses NLP techniques to recognize user queries and provide relevant responses.
- **AI/ML Learning Support**: Offers explanations, roadmaps, and resources for AI/ML concepts.
- **Conversation Logging**: Records interactions for users to revisit past discussions.
- **Streamlit Interface**: Simple and interactive UI for seamless engagement.

## How It Works
1. **Text Preprocessing**: Tokenization, stopword removal, and lemmatization for improved accuracy.
2. **Intent Classification**: TF-IDF vectorization + Logistic Regression for predicting user intent.
3. **Response Generation**: Retrieves a relevant answer from a predefined knowledge base.
4. **Logging Conversations**: Saves interactions in a CSV file for reference.

## Installation & Usage
### Prerequisites
Ensure you have **Python 3.x** and the required dependencies installed.

### Steps to Run
```sh
# Clone the repository
git clone https://github.com/Vedika-Sd/LearnAi-Chatbot.git
cd LearnAi-Chatbot

# Install dependencies
pip install -r requirements.txt

# Run the chatbot
streamlit run learnai.py
```

## Project Structure
```
LearnAi-Chatbot/
├── learnai.py          # Main chatbot script
├── intents.json        # Dataset with predefined intents & responses
├── chat_log.csv        # Logs user-chatbot interactions
├── requirements.txt    # Required dependencies
└── README.md           # Project documentation
```

## Limitations
- **Limited Context Awareness**: The chatbot does not remember past conversations.
- **Basic Model**: Uses Logistic Regression, which may not provide highly accurate responses.
- **Fixed Responses**: Responses are predefined and not dynamically generated.

## Future Improvements
- Implement **BERT/GPT-based** models for enhanced NLP capabilities.
- Add **context retention** to handle multi-turn conversations.
- Personalization with **user profiles** and adaptive learning recommendations.

## Contributing
Contributions are welcome! Feel free to fork the repo, raise issues, or submit pull requests.

## License
This project is open-source under the **MIT License**.

---
### 🚀 Start your AI/ML learning journey with LearnAI today! 🚀

###🚀 My First AI/ML Project
This chatbot marks the beginning of my AI/ML journey. As my first project, it helped me understand NLP, intent recognition, and basic ML models. While simple now, I plan to enhance it with advanced NLP techniques, contextual understanding, and dynamic recommendations in the future.


