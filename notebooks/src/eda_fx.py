from os import chmod, getcwd, makedirs, walk
from shutil import rmtree
import kaggle
import zipfile
from tqdm import tqdm
from os.path import expanduser, exists, join, dirname
from pandas import DataFrame
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa.display

import kaggle.api

from src.constants import EMOTIONS

def download_kaggle_dataset(dataset, output_folder="data/raw/"):

    kaggle_dir = expanduser('~/.kaggle')
    kaggle_json_path = join(kaggle_dir, 'kaggle.json')

    if not exists(kaggle_json_path):
        raise FileNotFoundError(f"Kaggle credentials file not found at {kaggle_json_path}. Please ensure it's placed there.")

    """with open(kaggle_json_path, 'r') as f:
        kaggle_creds = json.load(f)"""

    # Set permission for security
    chmod(kaggle_json_path, 0o600)

    output_folder = join(dirname(getcwd()), output_folder)

    if not exists(output_folder):
        makedirs(output_folder)

    # Kaggle directly uses the API credentials found in .kaggle/kaggle.json
    kaggle.api.dataset_download_files(dataset, path=output_folder, unzip=True)

    audio_folder = join(output_folder, "audio_speech_actors_01-24")
    
    if exists(audio_folder):
        rmtree(audio_folder)

    print(f"Dataset '{dataset}' downloaded and extracted to '{output_folder}'.")



def load_data(data_dir):
    data = []
    for root, _, files in walk(data_dir):
        for file in tqdm(files, disable=True):
            if file.endswith(".wav"):
                file_path = join(root, file)
                emotion = EMOTIONS[int(file.split('-')[2]) -1]
                data.append({
                    'path' : file_path,
                    'emotion' : emotion
                })
    return DataFrame(data)



def plot_emotion_distribution(df):
    fig = plt.figure(figsize=(12, 8))
    counts = df['emotion'].value_counts()
    bars = counts.plot(kind='bar', legend=False)

    plt.title('Emotions distribution')
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    for bar in bars.patches:
        plt.annotate(
            bar.get_height(),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center', 
            va='bottom'
        )

    total = counts.sum()
    plt.annotate(
        f'Total: {total}',
        xy=(0.97, 0.97), 
        xycoords='axes fraction',
        ha='right',
        va='top',
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightblue')
    )

    return fig




def waveplot(data, sr, emotion, play_audio = False):
    plt.figure(figsize=(10,6))
    plt.title(emotion.capitalize(), size= 20)
    librosa.display.waveshow(data, sr=sr)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    if play_audio:
        Audio(data=data, rate=sr)

def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(10,6))
    plt.title(emotion.capitalize(), size= 20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='log')   # log to see better low freq, otherwise hz
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()