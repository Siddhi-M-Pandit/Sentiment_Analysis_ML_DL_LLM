import numpy as np
import time
import os, csv
# point XLA at the NVVM libdevice bitcode
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda-12.3/nvvm/libdevice'

#os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from data_loader import MAX_VOCAB_SIZE, MAX_SEQ_LEN

CURRENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["MPLCONFIGDIR"] = "/tmp"

import yaml

import wandb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Load tokenized data splits (Only training & validation)
X_train_seq = np.load("data/splits/X_train_seq.npy")
X_val_seq = np.load("data/splits/X_val_seq.npy")

y_train_bin = np.load("data/splits/y_train_bin.npy")
y_val_bin = np.load("data/splits/y_val_bin.npy")

with open(os.path.join(CURRENT_DIR, "sweep_config.yaml")) as f:
    sweep_cfg = yaml.safe_load(f)

# Sweep initialization
def sweep_train():
    print("Running sweep_train()...")
    run = wandb.init()
    config = run.config

    run.config.update({
        "sweep_method": sweep_cfg["method"],
        "metric_name":  sweep_cfg["metric"]["name"],
        "metric_goal":  sweep_cfg["metric"]["goal"],
    })

    # Tuning
    start_tuning = time.time()

    model = Sequential()
    model.add(Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=config.embedding_dim, input_length=MAX_SEQ_LEN))

    if config.model_type == "CNN":
        model.add(Conv1D(filters=config.num_filters, kernel_size=config.kernel_size, activation='relu'))
        model.add(GlobalMaxPooling1D())
    elif config.model_type == "BiLSTM":
        model.add(Bidirectional(LSTM(units=config.lstm_units, dropout=config.dropout, recurrent_dropout=0.2, return_sequences=False)))

    model.add(Dense(config.dense_units, activation='relu'))
    model.add(Dropout(config.dropout))
    model.add(Dense(1, activation='sigmoid'))

    if config.optimizer.lower() == "adam":
        opt = Adam(learning_rate=config.learning_rate)
    elif config.optimizer.lower() == "rmsprop":
        opt = RMSprop(learning_rate=config.learning_rate)
    else:
        # fallback to whatever string they passed (won’t have custom LR)
        opt = config.optimizer

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    tuning_time = time.time() - start_tuning

    # Store tuning_time to csv file to reuse for total tuning_time later
    csv_path = os.path.join(os.getcwd(), "tuning_times.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # write header on first run
            writer.writerow(["run_id", "tuning_time"])
        writer.writerow([run.id, tuning_time])

    # Training
    start_train = time.time()

    model.fit(
        X_train_seq, np.array(y_train_bin),
        validation_data=(X_val_seq, np.array(y_val_bin)),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=0
    )

    train_time = time.time() - start_train


    # Evaluation
    print("carrying out validation eval..")
    y_val_pred = (model.predict(X_val_seq) > 0.5).astype("int32")

    val_accuracy = accuracy_score(y_val_bin, y_val_pred)
    val_precision = precision_score(y_val_bin, y_val_pred)
    val_recall = recall_score(y_val_bin, y_val_pred)
    val_f1 = f1_score(y_val_bin, y_val_pred)
    print("Done with validation eval!")


    # Log to W&B
    print("Logging metrics to W&B...")
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

# Launch Sweep
if __name__ == '__main__':
    # Load YAML config as a dictionary
    with open(os.path.join(CURRENT_DIR, "sweep_config.yaml")) as f:
        sweep_config = yaml.safe_load(f)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("GPU is available and memory growth is set!!! ")
        except RuntimeError as e:
            print("GPU initialization error:", e)
    else:
        print("No GPU available!!!! Check your setup...")

    sweep_train()
    
