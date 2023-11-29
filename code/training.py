import os
from matplotlib import pyplot
import librosa
from sklearn.preprocessing import LabelEncoder
from keras import utils as np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import numpy as np

def load_data(train_audio_path, target_sr=8000, test_size=0.2, random_state=42):
    all_wave = []
    all_label = []

    labels = os.listdir(train_audio_path)

    for label in labels:
        print(label)
        waves = [f for f in os.listdir(os.path.join(train_audio_path, label)) if f.endswith('.wav')]
        
        for wav in waves:
            try:
                # Carica e resampling
                y, sr = librosa.load(os.path.join(train_audio_path, label, wav), sr=target_sr)
                
                if len(y) == target_sr:
                    all_wave.append(y)
                    all_label.append(label)
            except Exception as e:
                print(f"Error loading {wav}: {e}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(all_label)
    y_categorical = np_utils.to_categorical(y_encoded, num_classes=len(labels))
    classes = list(le.classes_)

    all_wave = np.array(all_wave).reshape(-1, target_sr, 1)

    # set di addestramento e di validazione
    x_tr, x_val, y_tr, y_val = train_test_split(
        all_wave,
        y_categorical,
        stratify=y_categorical,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    return x_tr, x_val, y_tr, y_val, classes

def build_sequential_model():
    model = Sequential([
        Conv1D(filters=8, kernel_size=13, padding='valid', activation='relu', strides=1, input_shape=(8000, 1)),
        MaxPooling1D(3),
        Dropout(0.3),

        Conv1D(16, 11, padding='valid', activation='relu', strides=1),
        MaxPooling1D(3),
        Dropout(0.3),

        Conv1D(32, 9, padding='valid', activation='relu', strides=1),
        MaxPooling1D(3),
        Dropout(0.3),

        Conv1D(64, 7, padding='valid', activation='relu', strides=1),
        MaxPooling1D(3),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(256, activation='relu'),
        Dropout(0.3),

        Dense(len(labels), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    train_audio_path = "data\speech_commands_v0.01"

    x_tr, x_val, y_tr, y_val, labels = load_data(train_audio_path)

    K.clear_session()

    model = build_sequential_model()

    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001),
        ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    ]
    history = model.fit(
        x_tr, y_tr,
        epochs=100,
        callbacks=callbacks,
        batch_size=32,
        validation_data=(x_val, y_val),
        initial_epoch=0
    )

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    model.save("outputs/SpeechRecogModel.h5")
