import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
file_path = "sentiment-analysis.csv"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
cleaned_lines = [line.strip().strip('"') for line in lines]
cleaned_path = "sentiment-analysis-cleaned.csv"
with open(cleaned_path, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))
df = pd.read_csv(cleaned_path, sep=",", engine="python")
df.columns = ["Text", "Sentiment", "Source", "DateTime", "UserID", "Location", "Confidence"]
print("✅ Dataset Loaded Successfully")
print(df.head())
plt.figure(figsize=(6,4))
sns.countplot(x="Sentiment", data=df, palette="Set2")
plt.title("Sentiment Distribution")
plt.show()
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text
df["Clean_Text"] = df["Text"].astype(str).apply(clean_text)
X = df["Clean_Text"]
y = df["Sentiment"]
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
examples = ["This product is amazing, I love it!","Worst experience ever, totally disappointed.","Food was okay, not great but not bad either."]
example_tfidf = vectorizer.transform(examples)
print("\n Example Predictions:")
print(model.predict(example_tfidf))
