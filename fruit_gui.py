import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import json
import os

# -----------------------------
# Load model and class indices
# -----------------------------
MODEL_PATH = "fruit_classifier.h5"
CLASS_PATH = "class_indices.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_PATH):
    raise FileNotFoundError("❌ Please train the model first using train_model.py")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class index mapping
with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse dictionary {0: "Apple", 1: "Banana", ...}
class_labels = {v: k for k, v in class_indices.items()}


# -----------------------------
# Helper function: predict fruit
# -----------------------------
def predict_image(image_path):
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(100, 100))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions)) * 100

        fruit_name = class_labels[predicted_class]
        return fruit_name, confidence
    except Exception as e:
        return None, str(e)


# -----------------------------
# GUI App
# -----------------------------
class FruitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍎 Fruit Classifier")
        self.root.geometry("500x500")
        self.root.configure(bg="#f5f5f5")

        # Heading
        self.label = tk.Label(root, text="Fruit Classifier", font=("Arial", 20, "bold"), bg="#f5f5f5")
        self.label.pack(pady=10)

        # Image display
        self.image_label = tk.Label(root, bg="#f5f5f5")
        self.image_label.pack(pady=10)

        # Result text
        self.result_label = tk.Label(root, text="", font=("Arial", 16), bg="#f5f5f5", fg="green")
        self.result_label.pack(pady=10)

        # Buttons
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image,
                                    font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
        self.upload_btn.pack(pady=5)

        self.quit_btn = tk.Button(root, text="Quit", command=root.quit,
                                  font=("Arial", 12), bg="#f44336", fg="white", padx=10, pady=5)
        self.quit_btn.pack(pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if file_path:
            try:
                # Show image
                img = Image.open(file_path)
                img = img.resize((250, 250))
                img_tk = ImageTk.PhotoImage(img)

                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk

                # Predict
                fruit, confidence = predict_image(file_path)
                if fruit:
                    self.result_label.config(
                        text=f"Prediction: {fruit} ({confidence:.2f}%)", fg="blue"
                    )
                else:
                    self.result_label.config(
                        text=f"Error: {confidence}", fg="red"
                    )
            except Exception as e:
                messagebox.showerror("Error", f"Could not process image: {e}")


# -----------------------------
# Run GUI
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FruitApp(root)
    root.mainloop()
