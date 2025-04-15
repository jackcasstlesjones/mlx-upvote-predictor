# Hacker News Upvote Prediction Model: Data Analysis Report

## Key Findings

### Content Factors

- **Title Length**: Optimal range is 25-75 characters (13.5-16.5 avg score). Titles >100 characters perform significantly worse (5.5 avg score). Correlation is weakly negative (-0.01).

- **Domain Quality**: Highly specialised technical/educational domains consistently outperform others. Top domain (ciechanow.ski) achieves 444 avg score vs platform average of 14.

- **Content Type**: News domains slightly outperform non-news (18 vs 15 avg score).

### Author Factors

- **Author Experience**: Inverse relationship between post frequency and score for casual posters. Prolific posters (1000+ posts) show marginal improvement (15.7 avg score).

- **Karma Levels**: Mid-level karma users (100-500) achieve highest scores (32.3). Interestingly, high-karma authors (50,000+) perform below average (14.7).

- **Karma Impact**: Historical correlation between karma and score has declined steadily since 2010, now approaching zero.

### Timing Factors

- **Day of Week**: Weekend posts significantly outperform weekdays (Sunday: 16.2, Weekdays: ~13.0).

- **Time of Day**: Posts around noon/early afternoon perform best (14.5 avg score). Early morning posts (7-8am) perform worst (11.8).

- **Year Trends**: Average scores have increased steadily from 6.0 in 2006 to 17.3 in 2023, with a slight decline in 2024 (15.3).

## Recommendations for Model Features

### Must-Have Features

1. **Domain-Based**:

   - Domain reputation score (based on historical performance)
   - Dataset should be current (< 5 years old)

2. **Content-Based**:

   - Title length (with nonlinear encoding for 25-75 char sweet spot)
   - Title content (using word2vec)
   - Dataset should be current (< 5 years old)

3. **Timing-Based**:

   - Day of week (with weekend emphasis)
   - Hour of posting (with emphasis on noon-2pm peak)
   - Use all data available

## Deliberately Excluded

- **Author karma**: Despite conventional wisdom, shows very weak correlation (-0.02)
- **Author post frequency**: Minimal ßpredictive value (correlation 0.01)
- **Year trends**: While interesting historically, provides little predictive value for new posts

The data clearly shows domain quality and timing factors dominate prediction power, while author-based metrics have surprisingly little impact on outcomes.

The strongest predictor appears to be domain quality, followed by timing factors. Author karma - despite conventional wisdom - shows limited predictive value in recent data.
