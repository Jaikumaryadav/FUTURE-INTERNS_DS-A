📊 College Event Feedback – Data-Driven Insights in Action! 🎓✨

Just completed a full-cycle feedback analysis project using a powerful blend of Python and BI tools:

🧰 Tools Used:
🔹 Google Colab – for collaborative, cloud-based analysis
🔹 Pandas & CSV – to wrangle and structure student feedback
🔹 TextBlob & VADER – for sentiment analysis and polarity scoring
🔹 Seaborn – to explore trends and distributions
🔹 Power BI – to bring it all to life visually 🌟

📈 Highlights from the insights:
✅ 80%+ of students gave positive feedback
✅ Top appreciation for speakers, presentations, and event quality
✅ Key improvement area: time management
✅ Clear trends on how presentation quality links with subject expertise & doubt resolution

This project demonstrates how simple tools + thoughtful design can turn raw feedback into actionable insights — essential for improving academic and event experiences. 🚀



[CODE]


!pip install textblob nltk --quiet
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google.colab import files
uploaded = files.upload()  
df = pd.read_csv("student_feedback_updated.csv")

df.dropna(subset=["Event Feedback"], inplace=True)
df.reset_index(drop=True, inplace=True)

df[["Event Feedback"]].head()
df["Polarity_TextBlob"] = df["Event Feedback"].apply(lambda x: TextBlob(x).sentiment.polarity)

df["Sentiment_TextBlob"] = df["Polarity_TextBlob"].apply(
    lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral"
)

sid = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    score = sid.polarity_scores(text)
    compound = score["compound"]
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment_VADER"] = df["Event Feedback"].apply(get_vader_sentiment)
sns.countplot(data=df, x="Sentiment_VADER", palette="pastel")
plt.title("Sentiment Distribution (VADER)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.grid(True)
plt.show()
df.to_csv("analyzed_student_feedback.csv", index=False)
files.download("analyzed_student_feedback.csv")
