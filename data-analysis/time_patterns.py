import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.db_connection import get_cursor


def analyze_time_patterns():
    query = """
    SELECT time, score, title
    FROM hacker_news.items
    WHERE type = 'story'
    """

    with get_cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()

    df = pd.DataFrame(results, columns=['time', 'score', 'title'])

    # Check if time is already a datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        # If it's a Unix timestamp (integer), convert to datetime
        if pd.api.types.is_integer_dtype(df['time']):
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
        else:
            # If it's another format, try standard conversion
            df['datetime'] = pd.to_datetime(df['time'])
    else:
        # If it's already a datetime, just copy it
        df['datetime'] = df['time']

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour

    print("Data date range:", df['datetime'].min(), "to", df['datetime'].max())

    yearly_scores = df.groupby('year').agg({
        'score': ['mean', 'median', 'count']
    })
    yearly_scores.columns = ['avg_score', 'median_score', 'post_count']
    yearly_scores = yearly_scores.reset_index()

    print("\nScores by year:")
    print(yearly_scores)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='year', y='avg_score', data=yearly_scores, marker='o')
    plt.title('Average Score by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Score')

    plt.savefig('yearly_scores.png')
    print("Plot saved as 'yearly_scores.png'")

    hourly_scores = df.groupby('hour').agg({
        'score': ['mean', 'median', 'count']
    })
    hourly_scores.columns = ['avg_score', 'median_score', 'post_count']
    hourly_scores = hourly_scores.reset_index()

    print("\nScores by hour of day:")
    print(hourly_scores)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='hour', y='avg_score', data=hourly_scores, marker='o')
    plt.title('Average Score by Hour of Day')
    plt.xlabel('Hour (0-23)')
    plt.ylabel('Average Score')
    plt.xticks(range(0, 24))

    plt.savefig('hourly_scores.png')
    print("Plot saved as 'hourly_scores.png'")

    day_names = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_scores = df.groupby('day_of_week').agg({
        'score': ['mean', 'median', 'count']
    })
    daily_scores.columns = ['avg_score', 'median_score', 'post_count']
    daily_scores = daily_scores.reset_index()
    daily_scores['day_name'] = daily_scores['day_of_week'].apply(
        lambda x: day_names[int(x)])

    print("\nScores by day of week:")
    print(daily_scores)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='day_name', y='avg_score',
                data=daily_scores, order=day_names)
    plt.title('Average Score by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Score')

    plt.savefig('daily_scores.png')
    print("Plot saved as 'daily_scores.png'")

    recent_years = df[df['year'] >= df['year'].max() - 2]
    older_years = df[df['year'] < df['year'].max() - 2]

    recent_corr = calculate_correlations(recent_years)
    older_corr = calculate_correlations(older_years)

    print("\nCorrelations for recent posts (last 2 years):")
    print(recent_corr)

    print("\nCorrelations for older posts:")
    print(older_corr)


def calculate_correlations(df):
    title_length_corr = df['title'].str.len().corr(df['score'])

    hour_corr = pd.get_dummies(df['hour']).corrwith(df['score']).abs().mean()
    day_corr = pd.get_dummies(df['day_of_week']).corrwith(
        df['score']).abs().mean()

    return pd.Series({
        'title_length_correlation': title_length_corr,
        'hour_correlation': hour_corr,
        'day_of_week_correlation': day_corr
    })


if __name__ == "__main__":
    analyze_time_patterns()
