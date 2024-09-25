from librosa import load
from librosa.feature import mfcc, delta, chroma_stft, spectral_centroid, spectral_bandwidth, spectral_contrast,\
                        spectral_rolloff, zero_crossing_rate, rms, melspectrogram
from numpy import vstack, array, pad, mean, hstack
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder


def extract_features(file_path, duration=3, sr=16000, n_mfcc=20):
    # Load the audio file
    audio, sr = load(file_path, duration=duration, sr=sr)
    
    # Pad the audio signal to ensure it's of the correct length
    if len(audio) < sr * duration:
        audio = pad(audio, (0, sr * duration - len(audio)), 'constant')
    else:
        audio = audio[:sr * duration]  # Trim to max duration

    # Initialize a list to hold feature vectors
    features = []

    # Extract MFCC features and their deltas
    mfccs = mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_delta = delta(mfccs)
    mfccs_delta2 = delta(mfccs, order=2)

    # Append the MFCCs and their deltas to the feature list
    features.append(mean(mfccs, axis=1))
    features.append(mean(mfccs_delta, axis=1))
    features.append(mean(mfccs_delta2, axis=1))

    # Extract other features
    features.append(mean(chroma_stft(y=audio, sr=sr), axis=1))
    features.append(mean(spectral_centroid(y=audio, sr=sr), axis=1))
    features.append(mean(spectral_bandwidth(y=audio, sr=sr), axis=1))
    features.append(mean(spectral_contrast(y=audio, sr=sr), axis=1))
    features.append(mean(spectral_rolloff(y=audio, sr=sr), axis=1))
    features.append(mean(zero_crossing_rate(y=audio), axis=1))
    features.append(mean(rms(y=audio), axis=1))

    # Combine all features into a single feature vector
    feature_vector = hstack(features)

    return feature_vector


def preprocessing_df(df):

    X = array(df['features'].tolist())

    enc = OneHotEncoder()
    y = enc.fit_transform(df[['emotion']])

    return X, y.toarray()