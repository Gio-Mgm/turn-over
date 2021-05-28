from clearml import Task
import tensorflow as tf
from tensorflow.keras import callbacks
import autokeras as ak
import kerastuner as kt
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("./data/01_raw/attrition_train.csv")

df["Attrition"] = df["Attrition"].apply(lambda x: 0 if x == "No" else 1)
y = df["Attrition"]
X = df.drop('Attrition', axis=1)
X = pd.get_dummies(X)

# Initialize the structured data classifier.
model = ak.StructuredDataClassifier(
    overwrite=True,
    max_trials=1,
    objective="accuracy",    
    loss="binary_crossentropy"
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,  # minimium amount of change to count as an improvement
    patience=5,  # how many epochs to wait before stopping
    restore_best_weights=True,
)

model.fit(X, y, epochs=50, callbacks=[early_stopping])