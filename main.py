import tensorflow as tf
import numpy as np

import time
import json

# Load the data
with open("./data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_images = np.array([item["data"] for item in data]).reshape(-1, 3, 4, 1)
train_labels = np.array([item["label"] for item in data])

result = {}

for hiddenLayer in range(3):
    for Neuron in range(5):
        result[f"hiddenLayer_{hiddenLayer + 1}_Neuron_{Neuron + 1}"] = {}
        layers = [
            tf.keras.layers.Flatten(input_shape=(3, 4, 1)),
        ]
        for _ in range(hiddenLayer + 1):
            layers.append(tf.keras.layers.Dense(Neuron + 1, activation="relu"))
        layers.append(tf.keras.layers.Dense(2, activation="softmax"))
        model = tf.keras.Sequential(layers)

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        START = time.time()

        history = model.fit(train_images, train_labels, epochs=1000).history

        END = time.time()

        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

        potential_O_data = probability_model.predict(
            np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]).reshape(-1, 3, 4, 1)
        )
        potential_X_data = probability_model.predict(
            np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]).reshape(-1, 3, 4, 1)
        )
        result[f"hiddenLayer_{hiddenLayer + 1}_Neuron_{Neuron + 1}"]["O"] = {}
        result[f"hiddenLayer_{hiddenLayer + 1}_Neuron_{Neuron + 1}"]["O"]["t1"] = round(
            potential_O_data[0][0], 2
        )
        result[f"hiddenLayer_{hiddenLayer + 1}_Neuron_{Neuron + 1}"]["O"]["t2"] = round(
            potential_O_data[0][1], 2
        )

        result[f"hiddenLayer_{hiddenLayer + 1}_Neuron_{Neuron + 1}"]["X"] = {}
        result[f"hiddenLayer_{hiddenLayer + 1}_Neuron_{Neuron + 1}"]["X"]["t1"] = round(
            potential_X_data[0][0], 2
        )
        result[f"hiddenLayer_{hiddenLayer + 1}_Neuron_{Neuron + 1}"]["X"]["t2"] = round(
            potential_X_data[0][1], 2
        )
        result[f"hiddenLayer_{hiddenLayer + 1}_Neuron_{Neuron + 1}"]["Loss"] = round(
            history["loss"][-1], 2
        )
        result[f"hiddenLayer_{hiddenLayer + 1}_Neuron_{Neuron + 1}"]["Average_Loss"] = (
            round(sum(history["loss"]) / len(history["loss"]), 2)
        )
        result[f"hiddenLayer_{hiddenLayer + 1}_Neuron_{Neuron + 1}"]["Time"] = round(
            END - START, 2
        )

print(result)
