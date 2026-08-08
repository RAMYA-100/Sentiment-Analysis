# Sentiment Analysis and Filtering of Negative Feedback Using NLP
# Done By: Ramya L
# reg no: 212225040330
# Aim:
To perform sentiment analysis on social media text data using the VADER sentiment analyzer and filter only the negative feedback.
# Algorithm:
1.Load the Sentiment140 dataset using Pandas.

2.Initialize the VADER sentiment analyzer.

3.Analyze each text and calculate its compound sentiment score.

4.Identify records with a compound score less than 0.

5.Filter and display only the negative feedback.

6.Save the filtered data as negative_feedback.csv

# Program
```
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER
nltk.download("vader_lexicon")

# Column names for Sentiment140
columns = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "text"
]

# Read dataset
df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    names=columns,
    nrows=10000
)

print("Dataset loaded successfully!")
print("Total records:", len(df))

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Calculate sentiment score
df["compound_score"] = df["text"].astype(str).apply(
    lambda x: sid.polarity_scores(x)["compound"]
)

# Filter negative feedback
negative_df = df[df["compound_score"] < 0]

# Display negative feedback
print("\nNEGATIVE FEEDBACK")
print("=================")

print(
    negative_df[["text", "compound_score"]]
    .head(20)
    .to_string(index=False)
)

# Save results
negative_df[["text", "compound_score"]].to_csv(
    "negative_feedback.csv",
    index=False
)

print("\nNegative feedback count:", len(negative_df))
print("Saved as: negative_feedback.csv")
```
# Output
<img width="1497" height="747" alt="Screenshot 2026-08-08 093921" src="https://github.com/user-attachments/assets/5b693e04-4df6-4b50-a713-dfad8fc9fc45" />
