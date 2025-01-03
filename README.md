# Sentiment_analysis
This project focuses on Sentiment Analysis of social media posts, particularly tweets, using Python and Natural Language Processing (NLP) techniques. The goal is to classify tweets into Positive, Negative, and Neutral sentiments based on the textual content.
Key Features:

    Text Preprocessing: Cleaned and tokenized raw text by removing URLs, special characters, and stopwords, and applied lemmatization to normalize the data.
    Feature Extraction: Used CountVectorizer to convert cleaned text into numerical features for model training.
    Modeling: Trained a Logistic Regression model to predict sentiment categories.
    Visualization:
        Plotted the distribution of sentiment categories.
        Generated word clouds for each sentiment to visualize the most frequent words.
    Evaluation: Measured model performance using metrics such as accuracy, precision, recall, and F1-score.

Tools and Technologies Used:

    Programming Language: Python
    Libraries: Pandas, NumPy, NLTK, Scikit-learn, Matplotlib, WordCloud
    Dataset: A Kaggle dataset containing tweets with labeled sentiments (Positive, Negative, Neutral).

Outcomes:

    Successfully classified tweets with good accuracy.
    Provided insights into public sentiment trends and commonly used keywords in each sentiment category.
    Created reusable models and vectorizers for real-time sentiment prediction.

This project demonstrates the practical application of NLP and machine learning techniques for analyzing social media data, offering valuable insights for businesses and researchers.
