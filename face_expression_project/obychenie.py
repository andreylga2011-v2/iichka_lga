import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_dataset(folder_path):
    X = []
    y = []

    # Названия файлов должны быть вида: имя_метка.png (например: smile_1.png)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            label = int(filename.split('_')[-1].split('.')[0])  # smile_1.png → 1
            path = os.path.join(folder_path, filename)
            img = Image.open(path).convert('L').resize((8, 8))
            pixels = np.array(img).flatten()
            binary = (pixels < 128).astype(int)  # 1 = чёрный, 0 = белый
            X.append(binary)
            y.append(label + 1)  # Чтобы иметь классы 0, 1, 2 (для softmax)

    return np.array(X), to_categorical(y, num_classes=3)

def build_model():
    model = Sequential([
        Dense(32, input_shape=(64,), activation='relu'),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')  # 3 класса: грусть, нейтрально, улыбка
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(dataset_folder):
    X, y = load_dataset(dataset_folder)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model()
    model.fit(X_train, y_train, epochs=30, batch_size=4, validation_data=(X_test, y_test))
    model.save("face_expression_model.h5")
    print("Модель сохранена как face_expression_model.h5")

if __name__ == "__main__":
    train_model("dataset_faces")