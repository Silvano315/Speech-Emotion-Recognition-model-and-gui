from librosa import load
from librosa.feature import mfcc
from numpy import mean, array, pad
from sklearn.preprocessing import OneHotEncoder


def extract_features(file_path):
    audio, sr = load(file_path, duration=3, sr=22050)
    if len(audio) < sr * 3:
            audio = pad(audio, (0, sr * 3 - len(audio)))
    else:
        audio = audio[:sr * 3]
    mfcc_values = mfcc(y=audio, sr = sr, n_mfcc=40)
    return mean(mfcc_values.T, axis=0)


def preprocessing_df(df):

    X = array(df['features'].tolist())

    enc = OneHotEncoder()
    y = enc.fit_transform(df[['emotion']])

    return X, y