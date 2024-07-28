import tensorflow as tf
import numpy as np
import json

# Load the data
with open("./data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_images = np.array([item["data"] for item in data]).reshape(-1, 3, 4, 1)
train_labels = np.array([item["label"] for item in data])

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(3, 4, 1)),
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

model.fit(train_images, train_labels, epochs=1000)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# predictions_single = probability_model.predict(np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]).reshape(-1, 3, 4, 1))
predictions_single = probability_model.predict(np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]).reshape(-1, 3, 4, 1))

print(predictions_single)
