import io
import uuid
from pathlib import Path
from urllib.request import urlopen

import librosa
import numpy as np
from scipy import spatial

from spafe.features.lfcc import lfcc
from spafe.features.mfcc import mfcc
from spafe.features.pncc import pncc
from spafe.utils import vis

from resemblyzer import VoiceEncoder, preprocess_wav


class AudioFeaturizer():
    
    names_to_norm = ['lfcc', 'mfcc', 'pncc']
    path_to_temp_file = './temp.wav'
    feature_len = 384
    d_weight = 4
    lfcc_weight = 1.2
    pncc_weight = 0.5
    mfcc_weight = 0.5

    def __init__(self):
        self.encoder = VoiceEncoder()
    
    def read_file(self, file_name, sample_rate=16000):
        record, sample_rate = librosa.load(file_name, sr=sample_rate)
        return record, sample_rate

    def read_file_by_url(self, url, sample_rate=16000):
        z = io.BytesIO(urlopen(url).read())
        temp_path = f'./{str(uuid.uuid4())}.wav'
        Path(temp_path).write_bytes(z.getbuffer())
        record, sample_rate = librosa.load(temp_path, sr=sample_rate)
        Path.unlink(temp_path)
        # TODO: choose logic/add method for DTLN processing
        return record, sample_rate
    
    def norm_dim(self, features):
        return np.mean(features.T, axis=0)
    
    def get_d_vector(self, record, sample_rate):
        wav = preprocess_wav(record)

        embed = self.encoder.embed_utterance(wav)
        np.set_printoptions(precision=3, suppress=True)
        return embed
    
    def get_lfcc(self, record, sample_rate):
        return lfcc(record, fs=sample_rate)
        
    def get_mfcc(self, record, sample_rate):
        return mfcc(record, fs=sample_rate)
    
    def get_pncc(self, record, sample_rate):
        return pncc(record, fs=sample_rate)
    
    def get_all_features(self, record, sample_rate=16000, normalize_dim=False):
        # TODO: format every feature to similar structure
        all_features = {
            'lfcc': self.get_lfcc(record, sample_rate),
            'mfcc': self.get_mfcc(record, sample_rate),
            'pncc': self.get_pncc(record, sample_rate),
            'd_vector': self.get_d_vector(record)
        }
        
        if normalize_dim:
            for feature_name in all_features.keys():
                all_features[feature_name] = self.norm_dim(all_features[feature_name]) if self.is_in_norm_list(feature_name) else all_features[feature_name]
        
        return all_features

    def get_all_features_limited(self, record, sample_rate=16000):
        all_features = self.get_all_features(record, sample_rate, normalize_dim=True)
        for feature_name in all_features:
            if feature_name in self.names_to_norm:
                all_features[feature_name] = all_features[feature_name][:self.feature_len]
        return all_features
    
    def is_in_norm_list(self, feature_name):
        return any([name == feature_name for name in self.names_to_norm])
    
    def visualize_spectrogram(self, record, sample_rate):
        vis.spectogram(record, sample_rate)
        
    def visualize_features(self, 
                           features, 
                           feature_index='feature', frame_index='frame index', normalize_dim=False):
        if normalize_dim:
            vis.plot(features, feature_index, frame_index)
        else:
            vis.visualize_features(features, feature_index, frame_index)

    def cosine_similarity(self, x, y):
        return 1 - spatial.distance.cosine(x, y)

    def compare_two_features_sets(self, features_1, features_2):
        d_sim = self.cosine_similarity(features_1['d_vector'], features_2['d_vector'])
        lfcc_sim = self.cosine_similarity(features_1['lfcc'], features_2['lfcc'])
        pncc_sim = self.cosine_similarity(features_1['pncc'], features_2['pncc'])
        mfcc_sim = self.cosine_similarity(features_1['mfcc'], features_2['mfcc'])

        sum_sim = d_sim*self.d_weight + lfcc_sim*self.lfcc_weight + pncc_sim*self.pncc_weight + mfcc_sim*self.mfcc_weight
        # TODO: scale sim based on weights
        return sum_sim
        