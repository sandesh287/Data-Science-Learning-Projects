# 20. Detect emotion in Text using Natural Language ToolKit (NLTK)
# Basic sentiment analysis



# import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Download required NLTK data (only needed once)
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')


# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()


# Function to detect emotion in text
def detect_emotion(text):
  # Analyze sentiment
  scores = sid.polarity_scores(text)
  
  # Display sentiment scores
  print(f'\nSentiment Scores: {scores}')
  
  # Determine emotion based on scores
  if scores['compound'] >= 0.5:
    emotion = 'Joy'
  elif scores['compound'] <= -0.5:
    emotion = 'Sadness'
  elif scores['neg'] > 0.5:
    emotion = 'Anger'
  elif scores['neu'] > 0.7:
    emotion = 'Neutral'
  else:
    emotion = 'Mixed emotions'
  
  return emotion


# Sample text examples for emotion detection
text1 = """
I am so happy today! The weather is beautiful, and everything is going well. I feel very positive and motivated!
"""

text2 = """
Life is very sad and I want to run breaking into cards today.
"""


# Detect and print the emotion
emotion = detect_emotion(text1)
print(f'\nDetected Emotion: {emotion}')

emotion = detect_emotion(text2)
print(f'\nDetected Emotion: {emotion}')