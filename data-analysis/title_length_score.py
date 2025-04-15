import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.db_connection import get_cursor


def analyze_title_length_vs_score():
    query = """
    SELECT title, score
    FROM hacker_news.items
    WHERE type = 'story' AND title IS NOT NULL
    """

    with get_cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()

    df = pd.DataFrame(results, columns=['title', 'score'])
    df['title_length'] = df['title'].str.len()

    print(f"Total stories analyzed: {len(df)}")
    print(f"Average title length: {df['title_length'].mean():.2f} characters")
    print(f"Average score: {df['score'].mean():.2f}")

    correlation = df['title_length'].corr(df['score'])
    print(f"Correlation between title length and score: {correlation:.4f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='title_length', y='score', data=df, alpha=0.5)
    plt.title('Relationship Between Title Length and Score')
    plt.xlabel('Title Length (characters)')
    plt.ylabel('Score')

    plt.savefig('title_length_vs_score.png')
    print("Plot saved as 'title_length_vs_score.png'")

    bins = [0, 25, 50, 75, 100, 125, 150, 200]
    df['length_bin'] = pd.cut(df['title_length'], bins=bins)
    length_bins = df.groupby('length_bin').agg({
        'score': ['mean', 'median', 'count']
    }).reset_index()
    print("\nScores by title length bins:")
    print(length_bins)


if __name__ == "__main__":
    analyze_title_length_vs_score()
