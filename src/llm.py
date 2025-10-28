import io
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load game data
games_df = pd.read_csv("../data/games.csv")
desc_df = pd.read_csv("../data/game_descriptions.csv", encoding="utf-8-sig").rename(
    columns={"bgg_id": "BGGId", "full_description": "Description"}
)

# Merge datasets on BGGId
merged_df = pd.merge(
    games_df.drop(columns=["Description"], errors="ignore"),
    desc_df[["BGGId", "Description"]],
    on="BGGId",
    how="inner"
)

# Extract all category columns automatically
category_columns = [col for col in games_df.columns if col.startswith("Cat:")]

def get_llm_scores(user_description: str, min_players: int, category: str, top_n: int = 10):
    """
    Generate LLM-based relevance scores for candidate games based on the user description.
    Returns a DataFrame: [BGGId, Name, LLM_Score].
    """
    category_col = f"Cat:{category}"
    if category_col not in merged_df.columns:
        raise ValueError(f"Invalid category '{category}'. Available categories: {category_columns}")

    # Filter dataset by player count and category
    filtered_df = merged_df[
        (merged_df["MinPlayers"] <= min_players)
        & (merged_df["MaxPlayers"] >= min_players)
        & (merged_df[category_col] == 1)
    ]

    if filtered_df.empty:
        return pd.DataFrame(columns=["Name", "LLM_Score"])

    # Limit to top 20 by rating for token efficiency
    candidate_games = filtered_df.sort_values("AvgRating", ascending=False).head(20)

    # Prepare text for LLM input
    descriptions = "\n\n".join([
        f"Name: {row['Name']}\nYear: {row['YearPublished']}\nDescription: {row['Description']}"
        for _, row in candidate_games.iterrows()
    ])

    prompt = f"""
    The user described their ideal board game as follows:
    "{user_description}"

    You are given a list of candidate board games. 
    For each game, assign a relevance score between 0 and 1 that reflects how well it matches the user's description.
    Respond *only* in CSV format with two columns: Name, LLM_Score.
    Example:
    Name,LLM_Score
    Game A,0.92
    Game B,0.74

    Games:
    {descriptions}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert board game recommender that outputs structured data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    csv_output = response.choices[0].message.content.strip()
    csv_output = "\n".join(
        line for line in csv_output.splitlines() if not line.strip().startswith("```")
    ).strip()

    # Convert CSV text to DataFrame with resilient parsing
    try:
        llm_scores_df = pd.read_csv(io.StringIO(csv_output))
    except Exception:
        lines = [line for line in csv_output.splitlines() if "," in line]
        if not lines:
            return pd.DataFrame(columns=["Name", "LLM_Score"])
        rows = [line.split(",") for line in lines]
        header, data_rows = rows[0], rows[1:]
        if not data_rows:
            # treat single line response as data without header
            data_rows = [header]
            header = ["Name", "LLM_Score"][: len(data_rows[0])]
        llm_scores_df = pd.DataFrame(data_rows, columns=header)

    # Standardize column names and coerce expected schema
    llm_scores_df.columns = [col.strip().replace(" ", "_") for col in llm_scores_df.columns]
    if "LLM_Score" not in llm_scores_df.columns and len(llm_scores_df.columns) >= 2:
        llm_scores_df.rename(columns={llm_scores_df.columns[1]: "LLM_Score"}, inplace=True)
    if "Name" not in llm_scores_df.columns and len(llm_scores_df.columns) >= 1:
        llm_scores_df.rename(columns={llm_scores_df.columns[0]: "Name"}, inplace=True)

    for col, default in {"Name": "", "LLM_Score": pd.NA}.items():
        if col not in llm_scores_df.columns:
            llm_scores_df[col] = default
    llm_scores_df = llm_scores_df[["Name", "LLM_Score"]]
    # Coerce scores to numeric, salvaging simple textual annotations like "0.85 (high)"
    if llm_scores_df["LLM_Score"].dtype == object:
        llm_scores_df["LLM_Score"] = (
            llm_scores_df["LLM_Score"]
            .astype(str)
            .str.extract(r"([0-9]*\.?[0-9]+)")[0]
        )
    llm_scores_df["LLM_Score"] = pd.to_numeric(llm_scores_df["LLM_Score"], errors="coerce")
    llm_scores_df.dropna(subset=["LLM_Score"], inplace=True)
    if llm_scores_df.empty:
        raise ValueError(
            "LLM response did not contain any usable scores. "
            f"Raw response:\n{csv_output}"
        )
    llm_scores_df["LLM_Score"] = llm_scores_df["LLM_Score"].clip(0, 1)

    candidate_lookup = candidate_games[["BGGId", "Name"]].drop_duplicates()
    llm_scores_df = candidate_lookup.merge(llm_scores_df, on="Name", how="right")
    llm_scores_df.dropna(subset=["BGGId"], inplace=True)

    if top_n:
        llm_scores_df = llm_scores_df.sort_values("LLM_Score", ascending=False).head(top_n)

    return llm_scores_df[["BGGId", "Name", "LLM_Score"]]


if __name__ == "__main__":
    # Example usage
    df = get_llm_scores(
        user_description="I love cooperative adventure games with fantasy storytelling.",
        min_players=4,
        category="Strategy"
    )
    print(df)
