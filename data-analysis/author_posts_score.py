import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.db_connection import get_cursor


def analyze_author_activity_vs_score():
    query = """
    WITH author_post_count AS (
        SELECT
            "by" as author,
            COUNT(*) as post_count
        FROM hacker_news.items
        WHERE type = 'story' AND "by" IS NOT NULL
        GROUP BY "by"
    )

    SELECT
        i."by" as author,
        apc.post_count,
        AVG(i.score) as avg_score,
        COUNT(*) as stories
    FROM hacker_news.items i
    JOIN author_post_count apc ON i."by" = apc.author
    WHERE i.type = 'story'
    GROUP BY i."by", apc.post_count
    ORDER BY apc.post_count DESC
    """

    with get_cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()

    df = pd.DataFrame(results, columns=[
                      'author', 'post_count', 'avg_score', 'stories'])

    print(f"Total unique authors analyzed: {len(df)}")
    print(f"Average posts per author: {df['post_count'].mean():.2f}")
    print(f"Average score per author: {df['avg_score'].mean():.2f}")

    correlation = df['post_count'].corr(df['avg_score'])
    print(
        f"Correlation between post count and average score: {correlation:.4f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='post_count', y='avg_score', data=df,
                    size='stories', sizes=(20, 200), alpha=0.5)
    plt.title('Relationship Between Author Post Count and Average Score')
    plt.xlabel('Number of Posts by Author')
    plt.ylabel('Average Score')
    plt.xscale('log')

    plt.savefig('author_posts_vs_score.png')
    print("Plot saved as 'author_posts_vs_score.png'")

    bins = [1, 5, 10, 25, 50, 100, 500, 1000, df['post_count'].max()]
    df['post_count_bin'] = pd.cut(df['post_count'], bins=bins)
    post_bins = df.groupby('post_count_bin').agg({
        'avg_score': ['mean', 'median', 'count']
    }).reset_index()
    print("\nScores by author post count bins:")
    print(post_bins)

    top_authors = df.sort_values('avg_score', ascending=False).head(10)
    print("\nTop 10 authors by average score (with at least 5 posts):")
    print(top_authors[top_authors['stories'] >= 5])


if __name__ == "__main__":
    analyze_author_activity_vs_score()
