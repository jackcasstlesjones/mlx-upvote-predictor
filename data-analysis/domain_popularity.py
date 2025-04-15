import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from utils.db_connection import get_cursor


def analyze_domain_popularity():
    query = """
    SELECT url, score
    FROM hacker_news.items
    WHERE type = 'story' AND url IS NOT NULL
    """

    with get_cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()

    df = pd.DataFrame(results, columns=['url', 'score'])

    def extract_domain(url):
        try:
            domain = urlparse(url).netloc
            return domain.replace('www.', '')
        except (ValueError, AttributeError):
            return None

    df['domain'] = df['url'].apply(extract_domain)
    df = df.dropna(subset=['domain'])

    domain_stats = df.groupby('domain').agg({
        'score': ['mean', 'median', 'count']
    })

    domain_stats.columns = ['avg_score', 'median_score', 'post_count']
    domain_stats = domain_stats.reset_index()

    popular_domains = domain_stats[
        domain_stats['post_count'] >= 10
    ].sort_values(
        'avg_score', ascending=False
    )

    print(f"Total domains analyzed: {len(domain_stats)}")
    print(f"Domains with 10+ posts: {len(popular_domains)}")

    print("\nTop 20 domains by average score (with at least 10 posts):")
    print(popular_domains.head(20))

    plt.figure(figsize=(12, 8))
    top_plot = popular_domains.head(20).sort_values('avg_score')
    sns.barplot(x='avg_score', y='domain', data=top_plot)
    plt.title('Average Score by Domain (Top 20 Domains with 10+ Posts)')
    plt.xlabel('Average Score')
    plt.ylabel('Domain')

    plt.tight_layout()
    plt.savefig('domain_popularity.png')
    print("Plot saved as 'domain_popularity.png'")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='post_count', y='avg_score',
                    data=domain_stats, alpha=0.5)
    plt.xscale('log')
    plt.title('Domain Popularity vs Average Score')
    plt.xlabel('Number of Posts (log scale)')
    plt.ylabel('Average Score')

    plt.savefig('domain_count_vs_score.png')
    print("Plot saved as 'domain_count_vs_score.png'")

    news_domains = ['nytimes.com', 'cnn.com', 'bbc.co.uk', 'theguardian.com',
                    'washingtonpost.com', 'reuters.com', 'bloomberg.com']

    news_df = domain_stats[domain_stats['domain'].isin(news_domains)]
    non_news_df = domain_stats[~domain_stats['domain'].isin(
        news_domains) & (domain_stats['post_count'] >= 10)]

    print("\nComparison of official news sites vs other domains:")
    print(f"Average score for news sites: {news_df['avg_score'].mean():.2f}")
    print(
        f"Average score for other domains: "
        f"{non_news_df['avg_score'].mean():.2f}")


if __name__ == "__main__":
    analyze_domain_popularity()
