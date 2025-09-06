import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Parameters
# -----------------------------
img_size = (100, 100)
batch_size = 32
epochs = 12
data_dir = "fruits"   # dataset folder (must contain subfolders like Apple/, Banana/, etc.)

# -----------------------------
# Ensure dataset folder exists
# -----------------------------
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"❌ Dataset folder '{data_dir}' not found. "
                            "Make sure you have fruits/Apple, fruits/Banana, etc.")

# -----------------------------
# Data generators with augmentation
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,   # 80% train, 20% validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# -----------------------------
# Save class indices to JSON
# -----------------------------
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("✅ Saved class indices:", train_data.class_indices)

# -----------------------------
# Model definition
# -----------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(img_size[0], img_size[1], 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# -----------------------------
# Train safely
# -----------------------------
try:
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )
except Exception as e:
    print("❌ Training failed:", e)
    exit(1)

# -----------------------------
# Save the model
# -----------------------------
model.save("fruit_classifier.h5")
print("✅ Model saved as fruit_classifier.h5")
print("🎉 Training complete!")
