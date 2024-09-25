from librosa import load
from librosa.feature import mfcc, delta, chroma_stft, spectral_centroid, spectral_bandwidth, spectral_contrast,\
                        spectral_rolloff, zero_crossing_rate, rms, melspectrogram
from numpy import vstack, array, pad, mean, hstack
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder

import librosa
import numpy as np
from scipy.signal import resample
import random

def extract_features(file_path, n_mfcc=13, n_mels=40, n_fft=2048, hop_length=512):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    features = np.concatenate([mfccs, mel_spectrogram, chroma])
    return features

def extract_features_from_array(audio, sr, n_mfcc=13, n_mels=40, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    features = np.concatenate([mfccs, mel_spectrogram, chroma])
    return features

def time_stretch(audio, rate):
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, n_steps):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def add_noise(audio, noise_factor):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def data_augmentation(audio, sr):
    augmented = []
    augmented.append(time_stretch(audio, rate=random.uniform(0.8, 1.2)))
    augmented.append(pitch_shift(audio, sr, n_steps=random.uniform(-4, 4)))
    augmented.append(add_noise(audio, noise_factor=random.uniform(0.005, 0.02)))
    return augmented

def preprocess_and_augment(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features = extract_features(file_path)  # Extract features from the original audio
    augmented_audio = data_augmentation(audio, sr)  # Augment the audio
    # Extract features from the augmented audio arrays
    augmented_features = [extract_features_from_array(aug, sr) for aug in augmented_audio]
    return [features] + augmented_features
from sklearn.preprocessing import StandardScaler

def standardize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def preprocessing_df(df):

    X = array(df['features'].tolist())

    enc = OneHotEncoder()
    y = enc.fit_transform(df[['emotion']])

    return X, y.toarray()