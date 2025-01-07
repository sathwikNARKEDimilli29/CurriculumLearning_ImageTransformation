import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import shap


# Define paths and categories
attack_folder = [PATH_TO_ATTACK_TRAFFIC]
normal_folder = [PATH_TO_NORMAL_TRAFFIC]
categories = ["Normal", "Attack"]


# Function to load data
def load_data(folder_path, label):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file), nrows=7500)
            df["label"] = label
            data.append(df)
    return pd.concat(data, ignore_index=True)

# Define paths and categories
attack_folder = "[PATH_TO_ATTACK_DIRECTORY]"
normal_folder = "[PATH_TO_NORMAL_NETWORKDATA_DIRECTORY]"
categories = ["Normal", "Attack"]


# Function to load data
def load_data(folder_path, label):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(
                os.path.join(folder_path, file), nrows=7500
            )  # Remove nrows for actual model building
            df["label"] = label
            data.append(df)
    return pd.concat(data, ignore_index=True)


# Function to apply PCA to reduce numerical columns to 36
def apply_pca(data, n_components=36):
    numerical_columns = data.select_dtypes(include=["number"]).columns
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(data[numerical_columns])
    reduced_df = pd.DataFrame(
        reduced_features, columns=[f"PC{i+1}" for i in range(n_components)]
    )
    reduced_df["label"] = data["label"].values
    return reduced_df


# Function to generate images
def generate_images(data, output_folder, mode="gray"):
    os.makedirs(output_folder, exist_ok=True)
    numerical_columns = data.select_dtypes(include=["number"]).columns
    images = []
    labels = []
    for idx, row in data.iterrows():
        numerical_data = row[numerical_columns].values
        if len(numerical_data) < 36:
            continue
        normalized_data = (numerical_data - np.min(numerical_data)) / (
            np.max(numerical_data) - np.min(numerical_data) + 1e-5
        )
        image = normalized_data[:36].reshape((6, 6))
        if mode == "rgb":
            image = np.stack([image] * 3, axis=-1)
        else:
            image = image[..., np.newaxis]
        images.append(image)
        labels.append(row["label"])
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


# Load datasets
attack_data = load_data(attack_folder, 1)
normal_data = load_data(normal_folder, 0)

# Combine datasets
stage_1_data = normal_data
stage_2_data = attack_data[
    attack_data["Attack_type"].isin(
        ["OS_Fingerprinting", "Port_Scanning", "Vulnerability_scanner"]
    )
]
stage_3_data = attack_data[
    attack_data["Attack_type"].isin(["XSS", "SQL_injection", "Password", "Uploading"])
]
stage_4_data = attack_data[
    ~attack_data["Attack_type"].isin(
        [
            "OS_Fingerprinting",
            "Port_Scanning",
            "Vulnerability_scanner",
            "XSS",
            "SQL_injection",
            "Password",
            "Uploading",
        ]
    )
]

# Generate images for curriculum learning stages
stage_images = {}
stage_labels = {}

for i, (stage_name, stage_data) in enumerate(
    {
        "stage_1": stage_1_data,
        "stage_2": stage_2_data,
        "stage_3": stage_3_data,
        "stage_4": stage_4_data,
    }.items(),
    start=1,
):
    images_gray, labels_gray = generate_images(
        stage_data, f"./generated_images_stage_{i}_gray", mode="gray"
    )
    stage_images[f"{stage_name}_gray"] = images_gray
    stage_labels[f"{stage_name}_gray"] = labels_gray

# Neural Network Model with Attention, Self-Attention, Feedforward, and ResNet
input_layer = tf.keras.Input(shape=(36, 1))

# Adaptive Feature Mask Layer
adaptive_feature_mask = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(1, activation="sigmoid")
)(input_layer)

# Dynamic Convolutional Layer
conv_layer = tf.keras.layers.Conv1D(
    32, kernel_size=3, activation="relu", padding="same"
)(adaptive_feature_mask)

# Attention-based Temporal Encoder with Self-Attention
attention_layer = tf.keras.layers.Attention()([conv_layer, conv_layer])

# Residual connection and normalization
residual_1 = tf.keras.layers.Add()([conv_layer, attention_layer])
norm_1 = tf.keras.layers.LayerNormalization()(residual_1)

# First set of GRU and LSTM layers with self-attention
gru_layer_1 = tf.keras.layers.GRU(64, return_sequences=True)(norm_1)
lstm_layer_1 = tf.keras.layers.LSTM(32, return_sequences=True)(gru_layer_1)

# Self-attention on the output of the LSTM layer
self_attention_layer_1 = tf.keras.layers.Attention()([lstm_layer_1, lstm_layer_1])

# Residual connection and normalization
residual_2 = tf.keras.layers.Add()([lstm_layer_1, self_attention_layer_1])
norm_2 = tf.keras.layers.LayerNormalization()(residual_2)

# Second set of GRU and LSTM layers with self-attention
gru_layer_2 = tf.keras.layers.GRU(64, return_sequences=True)(norm_2)
lstm_layer_2 = tf.keras.layers.LSTM(32, return_sequences=True)(gru_layer_2)

# Self-attention on the output of the second LSTM layer
self_attention_layer_2 = tf.keras.layers.Attention()([lstm_layer_2, lstm_layer_2])

# Residual connection and normalization
residual_3 = tf.keras.layers.Add()([lstm_layer_2, self_attention_layer_2])
norm_3 = tf.keras.layers.LayerNormalization()(residual_3)

# Third set of GRU and LSTM layers with self-attention
gru_layer_3 = tf.keras.layers.GRU(64, return_sequences=True)(norm_3)
lstm_layer_3 = tf.keras.layers.LSTM(32, return_sequences=False)(gru_layer_3)

# Edge-optimized Quantization and Pruning
dropout_layer_1 = tf.keras.layers.Dropout(0.3)(lstm_layer_3)
dropout_layer_2 = tf.keras.layers.Dropout(0.3)(dropout_layer_1)

# Output Layer
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(dropout_layer_2)

# Compile Model
nn_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Print Model Summary
nn_model.summary()

# Integrate with existing pipeline
print("Training the Neural Network Model with Attention and ResNet ")
X_combined = np.vstack(
    [
        stage_images["stage_1_gray"],
        stage_images["stage_2_gray"],
        stage_images["stage_3_gray"],
        stage_images["stage_4_gray"],
    ]
)
y_combined = np.hstack(
    [
        stage_labels["stage_1_gray"],
        stage_labels["stage_2_gray"],
        stage_labels["stage_3_gray"],
        stage_labels["stage_4_gray"],
    ]
)

# Train-Test Split
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

# Reshape input for the NN model
X_train_nn = X_train_nn.reshape(-1, 36, 1)
X_test_nn = X_test_nn.reshape(-1, 36, 1)

# Train the Model
nn_model.fit(X_train_nn, y_train_nn, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the Model
nn_eval = nn_model.evaluate(X_test_nn, y_test_nn, verbose=1)


# Define CNN model
def create_cnn(input_shape):
    model = Sequential(
        [
            Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=input_shape
            ),
            Dropout(0.2),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            Dropout(0.3),
            MaxPooling2D((2, 2), padding="same"),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(len(categories), activation="softmax"),
        ]
    )
    return model


# Balance the dataset
def balance_dataset(images, labels):
    idx_class_0 = np.where(labels == 0)[0]
    idx_class_1 = np.where(labels == 1)[0]

    if len(idx_class_0) == 0 or len(idx_class_1) == 0:
        print("Cannot balance dataset: one of the classes is missing.")
        return images, labels

    min_samples = min(len(idx_class_0), len(idx_class_1))
    balanced_indices = np.hstack(
        [
            np.random.choice(idx_class_0, min_samples, replace=False),
            np.random.choice(idx_class_1, min_samples, replace=False),
        ]
    )
    return images[balanced_indices], labels[balanced_indices]


# Train and evaluate model for grayscale images only
for stage_name in ["stage_1_gray", "stage_2_gray", "stage_3_gray", "stage_4_gray"]:
    print(f"Training on {stage_name} data")
    images = stage_images[stage_name]
    labels = stage_labels[stage_name]

    if len(np.unique(labels)) < len(categories):
        print(f"Balancing dataset for {stage_name} due to lack of diversity.")
        images, labels = balance_dataset(images, labels)

    labels = to_categorical(labels, num_classes=len(categories))
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    model = create_cnn((6, 6, 1))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Handle classification report for missing classes
    all_classes = list(range(len(categories)))
    print(
        f"Classification Report for {stage_name} data:\n",
        classification_report(
            y_true_classes, y_pred_classes, labels=all_classes, target_names=categories
        ),
    )

# Explainable AI using SHAP
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:3])

# Visualize SHAP explanations
for i in range(3):
    shap.image_plot(shap_values, X_test[i : i + 1])


# Ensembling The NN model and the CNN model

import shap
import numpy as np

common_test_data = X_test_nn
nn_model_proba = nn_model.predict(common_test_data)
cnn_model_proba = model.predict(common_test_data.reshape(-1, 6, 6, 1))

if nn_model_proba.shape[1] == 1:
    nn_model_proba = np.hstack([1 - nn_model_proba, nn_model_proba])

# Step 2: Average the probabilities (Soft Voting)
ensemble_proba = (nn_model_proba + cnn_model_proba) / 2

# Step 3: Get final predictions
ensemble_predictions = np.argmax(ensemble_proba, axis=1)

# Step 4: Generate the classification report
ensemble_y_test = y_test_nn
print("Classification Report for Custom Ensembled Model:")
print(
    classification_report(
        ensemble_y_test, ensemble_predictions, target_names=categories
    )
)

# Use a subset of the test data for explanation
X_explain = X_test_nn[:10]

# Simplify the data by flattening it
X_explain_flat = X_explain.reshape(X_explain.shape[0], -1)


# Define a prediction function for SHAP
def predict_fn(data):
    reshaped_data = data.reshape(-1, 36, 1)
    return nn_model.predict(reshaped_data)


# Create a SHAP KernelExplainer
explainer = shap.KernelExplainer(predict_fn, X_explain_flat)

# Compute SHAP values
shap_values = explainer.shap_values(X_explain_flat)

# Visualize SHAP values for the first explanation
print("Visualizing SHAP explanations...")
shap.summary_plot(shap_values, X_explain_flat)
