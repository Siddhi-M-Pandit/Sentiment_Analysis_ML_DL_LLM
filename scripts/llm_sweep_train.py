#!/usr/bin/env python
# export XDG_CACHE_HOME=/tmp      # Redirects matplotlib, HF, and other apps
# export TRANSFORMERS_CACHE=/tmp/transformers    # HF transformer files
# export HF_HOME=/tmp/huggingface_cache          # HF hub metadata
# export MPLCONFIGDIR=/tmp/.matplotlib           # Matplotlib cache


import os, time, yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
#import tf_keras as keras
from tensorflow.keras.optimizers import Adam, RMSprop

import wandb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoConfig, AutoTokenizer, TFAutoModelForSequenceClassification

# Paths & Config
CURRENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Load sweep config
with open(os.path.join(CURRENT_DIR, "llm_sweep_config.yaml")) as f:
    sweep_cfg = yaml.safe_load(f)


#  GPU setup
# gpus = tf.config.list_physical_devices('GPU')
# if not gpus:
#     raise RuntimeError("No GPU found.")
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.debugging.set_log_device_placement(True) 

MAX_SEQ_LEN = 100

# Data Loader 
def load_splits():
    X_train = pd.read_csv("data/splits/X_train.csv")["clean_text"].tolist()
    X_val   = pd.read_csv("data/splits/X_val.csv")["clean_text"].tolist()

    label_map = {"Positive": 1, "Negative": 0}

    y_train = pd.read_csv("data/splits/y_train.csv")["sentiment_label"].map(label_map).tolist()
    y_val   = pd.read_csv("data/splits/y_val.csv")["sentiment_label"].map(label_map).tolist()

    return X_train, y_train, X_val, y_val  

# Tokenizer
def tokenize_dataset(tokenizer, texts, labels, max_len, batch_size):
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="tf"
    )
    ds = tf.data.Dataset.from_tensor_slices((dict(enc), labels))
    return ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)



# LLM Sweep Training Function
def sweep_train():

    run = wandb.init()
    cfg = run.config
    run.config.update({
        "sweep_method": sweep_cfg["method"],
        "metric_name":  sweep_cfg["metric"]["name"],
        "metric_goal":  sweep_cfg["metric"]["goal"]
    })

    # select model
    model_name = "prajjwal1/bert-tiny" if cfg.model_type=="BERT" else "sshleifer/tiny-distilroberta-base"

    hf_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=1,
        hidden_dropout_prob=cfg.dropout,
        attention_probs_dropout_prob=cfg.dropout
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=hf_config,
        from_pt=True
    )


    # Load data
    X_train, y_train, X_val, y_val = load_splits()             

    # tokenize data
    train_ds = tokenize_dataset(tokenizer, X_train, y_train, MAX_SEQ_LEN, cfg.batch_size)
    val_ds   = tokenize_dataset(tokenizer, X_val,   y_val,   MAX_SEQ_LEN, cfg.batch_size)


    # compile
    start_tuning = time.time()

    if cfg.optimizer.lower() == "adam":
        opt = Adam(learning_rate=cfg.learning_rate)
    elif cfg.optimizer.lower() == "rmsprop":
        opt = RMSprop(learning_rate=cfg.learning_rate)

    loss_fn   = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

    tuning_time = time.time() - start_tuning


    # Training
    start_train = time.time() 

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=cfg.epochs,
                        #callbacks=[EarlyStopping(patience=1, restore_best_weights=True)],
                        verbose=0) 
    
    train_time = time.time() - start_train


    # validation eval
    print("carrying out validation eval..")

    logits = model.predict(val_ds).logits
    y_pred = (tf.sigmoid(logits).numpy() > 0.5).astype(int).flatten()

    val_accuracy  = accuracy_score(y_val, y_pred)
    val_precision = precision_score(y_val, y_pred)
    val_recall    = recall_score(y_val, y_pred)
    val_f1        = f1_score(y_val, y_pred)

    print("Done with validation eval!")


    # Log to W&B
    wandb.log({
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,

        "tuning_time": tuning_time,
        "training_time": train_time,

        "sweep_method": run.config.sweep_method,
        "metric_name": run.config.metric_name,
        "metric_goal": run.config.metric_goal,
    })

    print("Done logging metrics to W&B!")
    run.finish()



if __name__ == "__main__":
    # Load sweep config
    with open(os.path.join(CURRENT_DIR, "llm_sweep_config.yaml")) as f:
        sweep_cfg = yaml.safe_load(f)

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_memory_growth(gpus[0], True)
    #         print("GPU is available and memory growth is set!!! ")
    #     except RuntimeError as e:
    #         print("GPU initialization error:", e)
    # else:
    #     print("No GPU available!!!! Check your setup...")

    # run sweeps
    sweep_train()