import os
import librosa
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

from sklearn.preprocessing import LabelEncoder

def classMake(train_audio_path):
    labels = os.listdir(train_audio_path)
    all_wave = []
    all_label = []
    for label in labels:
        print(label)
        waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
        for wav in waves:
            # Load example trumpet signal
            y, sr = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
            # Resample to the target sampling rate (e.g., 8000 Hz)
            samples = librosa.resample(y, orig_sr=sr, target_sr=8000)

            if(len(samples)== 8000) : 
                all_wave.append(samples)
                all_label.append(label)

    le = LabelEncoder()
    y=le.fit_transform(all_label)
    classes= list(le.classes_)
    # SAVE CLASSES
    np.save('outputs/classes.npy', classes)

def create_audio(filename):
    samplerate = 16000  # Hertz

    print("Press Enter to start recording. Press Ctrl+C to stop recording.")

    try:
        input("Press Enter to start recording...")
        mydata = sd.rec(int(samplerate), samplerate=samplerate, channels=1, dtype=np.int16, blocking=True)
        print("Recording... Press Ctrl+C to stop.")
        while True:
            block = sd.rec(int(samplerate), samplerate=samplerate, channels=1, dtype=np.int16, blocking=True)
            mydata = np.concatenate((mydata, block), axis=None)
    except KeyboardInterrupt:
        print("\nRecording stopped.")

    sf.write(filename, mydata, samplerate)

def plot_wave(filename):    
    samplerate = 16000  # Hertz
    audio, _ = sf.read(filename, dtype=np.float32)

    # Plot the audio waveform
    time = np.arange(0, len(audio)) / samplerate
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio)
    plt.title('Audio Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.show()

def predict(audio, model):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return CLASSES[index]

def predict_audio(filename, model_path, tot=0.02):
    model = load_model(model_path)
    audio, _ = sf.read(filename, dtype=np.float32)
    start_idx = 0

    slice_size = 8000  # Adjust this size based on your model's input size
    i = 0
    while i < len(audio):

        if np.abs(audio[i]) > 0.02:
            j = i+7000
            while np.mean(np.abs(audio[j:j+200])) > 0.01:
                j = j + 1

            audio_slice = audio[i-200:j+400]
            sf.write(f"outputs/slice{i}.wav", audio_slice, 16000)
            print(len(audio_slice))
            audio_slice = np.resize(audio_slice, (slice_size,))
            for k in audio_slice:
                if(k > 0.02):
                    
                    print(predict(audio_slice, model))
                    break
            i = j+400
            print(i)
        else:
            i = i+1


if __name__ == "__main__":
    model_path = "outputs/SpeechRecogModel.h5"
    audio_file_path = "outputs/yes.wav"

    train_audio_path = "data\speech_commands_v0.01"
    classMake(train_audio_path) # has to be done just once

    CLASSES = np.load('outputs/classes.npy')
    create_audio(audio_file_path)
    # Make predictions
    predict_audio(audio_file_path, model_path)

    
    plot_wave(audio_file_path)