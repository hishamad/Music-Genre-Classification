import librosa
import numpy as np
import pandas as pd
import ast
from scipy.stats import skew, kurtosis


def calc_stats(feature):
        return np.hstack([
            np.mean(feature), np.var(feature), np.std(feature),
            np.median(feature), np.max(feature), 
            np.min(feature), skew(feature), 
            kurtosis(feature),
        ])

def extract_features(audio_file):

        # Load the audio file using librosa
        y, sr = librosa.load(audio_file, sr=None, mono=True)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        #alla 20 på mfcc
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        chroma_stats = []
        for i in range(chroma_stft.shape[0]):
            chroma_stats.append(calc_stats(chroma_stft[i, :]))  
        chroma_stats = np.hstack(chroma_stats)
        #print(f"Chroma STFT Shape: {chroma_stft.shape}")
        #print(f"Chroma STFT stats Shape: {chroma_stats.shape}")

        spectral_centroid_stats = calc_stats(np.hstack(spectral_centroid))
        #print(f"Spectral Centroid  Shape: {spectral_centroid.shape}")
        #print(f"Spectral Centroid stats  Shape: {spectral_centroid_stats.shape}")


        spectral_contrast_stats = []
        for i in range(spectral_contrast.shape[0]):
            spectral_contrast_stats.append(calc_stats(spectral_contrast[i, :])) 
        spectral_contrast_stats = np.hstack(spectral_contrast_stats)
        #print(f"Spectral Contrast Shape: {spectral_contrast.shape}")
        #print(f"Spectral Contrast stats Shape: {spectral_contrast_stats.shape}")

        zero_crossing_stats = calc_stats(np.hstack(zero_crossing_rate))
        #print(f"Zero Crossing Rate Shape: {zero_crossing_rate.shape}")
        #print(f"Zero Crossing Rate stast Shape: {zero_crossing_stats.shape}")


        mfcc_stats = []
        for i in range(mfccs.shape[0]):
            mfcc_stats.append(calc_stats(mfccs[i, :]))  

        mfcc_stats = np.hstack(mfcc_stats)
        #print(f"MFCC Stats Shape: {mfccs.shape}")
        #print(f"MFCC Stats stats Shape: {mfcc_stats.shape}")

        

        

        
        # Aggregate each feature into mean and variance
        features =np.hstack( [
            chroma_stats,
            spectral_centroid_stats,
            spectral_contrast_stats,
            zero_crossing_stats,
            mfcc_stats])

      
        #print(f"Final Features Shape: {features.shape}")
        return np.array(features)

  


# The code from fma https://github.com/mdeff/fma/blob/master/utils.py#L183
def load_tracks(filepath):
    if 'tracks' in filepath:
        # Load tracks.csv
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        
        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

       
        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks

'''
import time
X = []
y = []

# Extract features for each track in fma_small
for idx, row in fma_small.iterrows():
    #start_time = time.time()

    track_id = row.name 
    genre_label = row[('track', 'genre_top')]  
    
    # Construct the file path
    directory = '{:03d}'.format(track_id // 1000)
    filename = '{:06d}.mp3'.format(track_id)
    file_path = os.path.join(audio_dir, directory, filename)
    
    try:
        features = extract_features(file_path)
        X.append(features)
        y.append(genre_label)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    if idx % 1000 == 0 and idx != 0:
            print(f"Processed {idx} files so far")
    #elapsed_time = time.time() - start_time
    #print(elapsed_time)


# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

np.save('X.npy', X)
np.save('y.npy', y)
 '''