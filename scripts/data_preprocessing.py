import re
import pandas as pd

"""
Function to clean the tweet text 
"""
def clean_text(text):
    
    text = text.lower()                             # lowercase
    text = re.sub(r'http\S+|www.\S+', '', text)     # remove URLs
    text = re.sub(r'@\w+', '', text)                # remove usernames
    text = re.sub(r'#\w+', '', text)                # remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)         # remove special characters, numbers
    text = re.sub(r'\s+', ' ', text).strip()        # remove extra spaces

    return text


"""
Load and preprocess sentiment140_dataset.csv
"""
def preprocess_social_dataset(path):
    
    print(f"Loading data from: {path}")

    df = pd.read_csv(path, encoding='latin-1', header=None)
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    
    # Keeping only Positive & Negative classes 
    df = df[df['target'].isin([0, 4])]
    
    # Map target labels
    df['sentiment_label'] = df['target'].map({0: 'Negative', 4: 'Positive'})
    
    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    
    return df[['sentiment_label', 'text', 'clean_text']]
