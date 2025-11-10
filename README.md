# Board Game Recommender System

A hybrid recommendation system and interactive web app that suggests new board games based on player preferences, leveraging data from [BoardGameGeek](https://boardgamegeek.com/). Built with **Streamlit**, **Python**, and **machine learning**, this project demonstrates applied knowledge of Collaborative Filtering, Content-Based Filtering, and Large Language Models (LLMs).

## 🧰 Setup
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. In your project root, create a folder named `.streamlit` (if it doesn’t exist), then create a file inside it called `secrets.toml`:
```bash
.streamlit/secrets.toml
```
Add your OpenAI API key inside:
```toml
OPENAI_API_KEY = "your_api_key_here"
```

## 🚀 Run the App
From the project root:
```bash
streamlit run src/app.py
```

## 👩‍💻 Authors

**Team 43 — Georgia Tech**  
Chrissa da Gomez, Rene Pirolt, Bill Dvorkin, Evan Kai Hallberg, Elizabeth Kirk
