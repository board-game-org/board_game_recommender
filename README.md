# Board Game Recommender System

A data visualization and recommendation project that uses data from [BoardGameGeek.com](https://boardgamegeek.com/) to suggest new board games to users based on their preferences and game attributes.


## 📊 Data Source
Dataset from Kaggle: [Board Games Database from BoardGameGeek](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek/data)

## 🧰 Requirements

Make sure you have **Python 3.8+** installed. Install dependencies using:

```bash
pip install -r requirements.txt
```

If you don’t have Jupyter installed yet:
```bash
pip install jupyter
```
Make sure to have a data folder consisting of:
- data

## How to Run the App
1. Make sure OPENAI_API_KEY value is replaced in .streamlit/secrets.toml.
2. Run from root workspace: `streamlit run src/app.py`

## ▶️ How to Run the Jupyter Notebook

Activate your virtual environment (if using one):
```bash
source venv/bin/activate       # macOS/Linux
.\venv\Scripts\activate        # Windows
```

Launch Jupyter Notebook:
```bash
jupyter notebook
```

Once Jupyter opens in your browser, navigate to the project folder and open a notebook file. For example:
```bash
notebooks/data_exploration.ipynb
```

## 🧱 Build Collaborative Filtering Dataset
You can generate a compact dataset for user-based Collaborative Filtering using the new script:
Script: `src/build_cf_dataset.py`
Purpose: Combines user_ratings.csv with games.csv to produce a minimal ratings matrix with three columns:
`user_id`, `game_id`, `rating`

Output
data/cf_dataset.csv with columns:
- user_id (string or int from user ratings)
- game_id (matches BGGId in games.csv)
- rating (numeric user rating)

## 💡 Notes

- Running the Jupyter Notebook will generate a new file called `games_full.csv` in the `data/` directory.
- Large CSV files should be stored inside the `data/` directory (not committed to GitHub).

## 👩‍💻 Authors

**Team 43 — Georgia Tech**  
Chrissa da Gomez, Rene Pirolt, Bill Dvorkin, Evan Kai Hallberg, Elizabeth Kirk