# Speech Recognition using Convolutional Neural Networks
This repository contains code for a speech recognition system implemented using Convolutional Neural Networks (CNNs). The system is trained on the Speech Commands Dataset to recognize spoken words.

## Project Structure
The project is organized as follows:

1. data/: This directory contains the Speech Commands Dataset. Ensure that the dataset is downloaded and placed in this directory.
        In order to download the dataset, run the following command:
```bash
kaggle competitions download -c tensorflow-speech-recognition-challenge

unzip tensorflow-speech-recognition-challenge.zip -d data/
```
        Be also sure to remove any additional file from the data directory (e.g. the zip file, the csv file, readme and background noise folder)

2. outputs/: This directory is used to store the trained model and other outputs generated during the process.

3. code/: The source code for the project is organized in this directory. It includes the following main files:

4. training.py: This script loads the dataset, preprocesses the audio data, builds and trains the CNN model, and saves the trained model.

5. prova_del_9.py: This script contains functions for recording audio, predicting spoken words using the trained model, and visualizing the recorded audio waveform.

5. Preprocessing_creazione_DS.ipynb: This file contains utility functions used for data loading, preprocessing, and some exploratory data analysis.

## How to Use
1. Clone the repository:
    
```bash
git clone https://github.com/stecno04/speech-recognition-cnn.git

cd speech-recognition-cnn
```

2. Download the Speech Commands Dataset and place it in the data/ directory.

3. Install the required dependencies:
    
```bash
pip install -r requirements.txt
```

4. Train the model:

```bash
python code/training.py
```
This script will preprocess the data, build and train the CNN model, and save the trained model in the outputs/ directory.


Record and predict:

```bash

python code/prova_del_9.py
```
This script will record a short audio clip, use the trained model to predict the spoken word, and display the audio waveform.

