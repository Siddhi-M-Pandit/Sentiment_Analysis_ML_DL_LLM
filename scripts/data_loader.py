# For Tokenization, Binary Labelling & Padding the data splits

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_VOCAB_SIZE = 10000
MAX_SEQ_LEN = 100

def save_tokenize_data():

    # Load preprocessed data splits 
    X_train = pd.read_csv("data/splits/X_train.csv")["clean_text"].tolist()
    X_val = pd.read_csv("data/splits/X_val.csv")["clean_text"].tolist()
    X_test = pd.read_csv("data/splits/X_test.csv")["clean_text"].tolist()

    y_train = pd.read_csv("data/splits/y_train.csv")["sentiment_label"].tolist()
    y_val = pd.read_csv("data/splits/y_val.csv")["sentiment_label"].tolist()
    y_test = pd.read_csv("data/splits/y_test.csv")["sentiment_label"].tolist()


    # Binary labels: 1 for Positive, 0 for Negative
    y_train_bin = [1 if y == "Positive" else 0 for y in y_train]
    y_val_bin = [1 if y == "Positive" else 0 for y in y_val]
    y_test_bin = [1 if y == "Positive" else 0 for y in y_test]

    # Tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    # Padding
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_SEQ_LEN, padding='post')
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=MAX_SEQ_LEN, padding='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_SEQ_LEN, padding='post')

    # Save tokenized & padded splits
    np.save("data/splits/X_train_seq.npy", X_train_seq)
    np.save("data/splits/X_val_seq.npy", X_val_seq)
    np.save("data/splits/X_test_seq.npy", X_test_seq)
    np.save("data/splits/y_train_bin.npy", y_train_bin)
    np.save("data/splits/y_val_bin.npy", y_val_bin)
    np.save("data/splits/y_test_bin.npy", y_test_bin)

    print("Tokenized & padded data splits saved!")

if __name__ == '__main__':
    save_tokenize_data()