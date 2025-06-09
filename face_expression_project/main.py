import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime

LABELS = ['Грусть', 'Нейтрально', 'Улыбка']
LOG_FILENAME = "log.txt"


# ----------------------- Обработка данных -----------------------

def load_dataset(folder_path):
    X = []
    y = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                label_str = os.path.splitext(filename)[0].split('_')[-1]
                label = int(label_str)
                path = os.path.join(folder_path, filename)
                img = Image.open(path).convert('L').resize((8, 8))
                pixels = np.array(img).flatten() / 255.0
                X.append(pixels)
                y.append(label + 1)
            except Exception as e:
                print(f"Пропущен файл {filename}: {e}")

    return np.array(X), tf.keras.utils.to_categorical(y, num_classes=3)


def preprocess_image(path):
    img = Image.open(path).convert('L').resize((8, 8))
    pixels = np.array(img).flatten() / 255.0
    return pixels


# ----------------------- Модель -----------------------

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, input_shape=(64,), activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model():
    if not os.path.exists("dataset_faces"):
        messagebox.showerror("Ошибка", "Папка dataset_faces не найдена.")
        return

    X, y = load_dataset("dataset_faces")
    if X.shape[0] == 0:
        messagebox.showerror("Ошибка", "Не удалось загрузить данные для обучения.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = build_model()
    model.fit(X_train, y_train, epochs=30, batch_size=4, validation_data=(X_test, y_test))

    model_name = f"face_expression_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    model.save(model_name)

    loss, acc = model.evaluate(X_test, y_test)
    messagebox.showinfo("Готово", f"Модель сохранена как: {model_name}\nТочность на тесте: {acc:.2f}")


# ----------------------- Лог -----------------------

def log_prediction(image_name, label, probs):
    with open(LOG_FILENAME, "a", encoding='utf-8') as f:
        f.write(f"{datetime.now()} | {image_name} | {label} | {np.round(probs, 2).tolist()}\n")


def read_log():
    if not os.path.exists(LOG_FILENAME):
        return "Лог пуст."
    with open(LOG_FILENAME, "r", encoding='utf-8') as f:
        return f.read()[-1000:]


def clear_log():
    if os.path.exists(LOG_FILENAME):
        os.remove(LOG_FILENAME)
        messagebox.showinfo("Успешно", "Лог очищен.")


# ----------------------- GUI -----------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание эмоций (8x8)")
        self.image_path = None

        tk.Button(root, text="1. Обучить модель", command=train_model).pack(pady=5)
        tk.Button(root, text="2. Выбрать изображение", command=self.choose_image).pack(pady=5)
        tk.Button(root, text="3. Предсказать эмоцию", command=self.predict).pack(pady=5)
        tk.Button(root, text="4. Показать лог", command=self.show_log).pack(pady=5)
        tk.Button(root, text="5. Очистить лог", command=clear_log).pack(pady=5)

        self.canvas = tk.Label(root)
        self.canvas.pack()
        self.result_label = tk.Label(root, text="", font=('Arial', 14))
        self.result_label.pack(pady=5)

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Изображения", "*.png *.jpg *.jpeg")])
        if path:
            self.image_path = path
            img = Image.open(path).resize((80, 80))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.configure(image=img_tk)
            self.canvas.image = img_tk
            self.result_label.config(text="")

    def predict(self):
        if not self.image_path:
            messagebox.showwarning("Внимание", "Сначала выберите изображение.")
            return

        latest_model = self.find_latest_model()
        if not latest_model:
            messagebox.showerror("Ошибка", "Сначала обучите модель.")
            return

        model = tf.keras.models.load_model(latest_model)
        processed = preprocess_image(self.image_path)
        prediction = model.predict(np.array(processed).reshape(1, -1))[0]
        class_index = np.argmax(prediction)
        label = LABELS[class_index]
        probs = np.round(prediction, 2)

        self.result_label.config(text=f"Результат: {label} (вероятности: {probs})")
        log_prediction(os.path.basename(self.image_path), label, probs)

    def find_latest_model(self):
        files = [f for f in os.listdir() if f.startswith("face_expression_model_") and f.endswith(".h5")]
        if not files:
            return None
        return max(files, key=os.path.getctime)

    def show_log(self):
        log_text = read_log()
        top = tk.Toplevel(self.root)
        top.title("Последние записи лога")
        text_widget = tk.Text(top, wrap='word', height=20, width=80)
        text_widget.insert('1.0', log_text)
        text_widget.pack()


# ----------------------- Запуск -----------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

