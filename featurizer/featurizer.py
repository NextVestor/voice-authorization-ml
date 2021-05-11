import io
import math
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
    """AudioFeaturizer allows to get features from audio records.

    This class allows to read audio record as file or by URL, get features
    from audio record and compare different sets of features to get similarity. 

    :param names_to_norm: Names of features that should be normilized.
    :type names_to_norm: list
    :param path_to_temp_file: File name of temporary saved audio record.
    :type path_to_temp_file: str
    :param feature_len: Default lenght of feature vector.
    :type feature_len: int
    :param d_weight: Weight for d-vector feature, which is used in process of comparison.
    :type d_weight: float
    :param lfcc_weight: Weight for lfcc feature, which is used in process of comparison.
    :type lfcc_weight: float
    :param pncc_weight: Weight for pncc feature, which is used in process of comparison.
    :type pncc_weight: float
    :param mfcc_weight: Weight for mfcc feature, which is used in process of comparison.
    :type mfcc_weight: float
    """

    names_to_norm = ['lfcc', 'mfcc', 'pncc']
    path_to_temp_file = './temp.wav'
    feature_len = 384
    d_weight = 4.0
    lfcc_weight = 1.2
    pncc_weight = 0.5
    mfcc_weight = 0.5

    def __init__(self):
        """At initialization level AudioFeaturizer loads VoiceEncoder model.
        """
        #: VoiceEncoder: VoiceEncoder that used to get d-vector from audio record.
        self.encoder = VoiceEncoder()
    
    def read_file(self, file_name, sample_rate=16000):
        """Read audio record as file.
            
        :param file_name: Path to the audio file.
        :type file_name: str
        :param sample_rate: Sample rate for audio record.
        :type sample_rate: int

        :return: Record and sample rate
        """
        record, sample_rate = librosa.load(file_name, sr=sample_rate)
        return record, sample_rate

    def read_file_by_url(self, url, sample_rate=16000):
        """Read audio record by URL.
            
        :param url: URL to the audio file.
        :type url: str
        :param sample_rate: Sample rate for audio record.
        :type sample_rate: int

        :return: Record and sample rate
        """
        z = io.BytesIO(urlopen(url).read())
        temp_path = f'./{str(uuid.uuid4())}.wav'
        temp_file = Path(temp_path).write_bytes(z.getbuffer())
        record, sample_rate = librosa.load(temp_path, sr=sample_rate)
        Path(temp_path).unlink()
        return record, sample_rate
    
    def norm_dim(self, features):
        """Normalize feature vector to 1-dimensional with default lenght.
            
        :param features: Feature vector.
        :type features: list

        :return: Normalized features
        """
        return np.mean(features.T, axis=0)
    
    def get_d_vector(self, record, sample_rate):
        """Get d-vector feature from audio record.
            
        Args:
        :param record: Record object to get feature from.
        :type record: object
        :param sample_rate: Sample rate for audio record.
        :type sample_rate: int

        :return: D-vector feature vector
        """
        wav = preprocess_wav(record)

        embed = self.encoder.embed_utterance(wav)
        np.set_printoptions(precision=3, suppress=True)
        return embed
    
    def get_lfcc(self, record, sample_rate):
        """Get LFCC feature from audio record.
            
        `LFCC paper <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.8029&rep=rep1&type=pdf>`_
        
        :param record: Record object to get feature from.
        :type record: object
        :param sample_rate: Sample rate for audio record.
        :type sample_rate: int

        :return: LFCC feature vector
        """
        return lfcc(record, fs=sample_rate)
        
    def get_mfcc(self, record, sample_rate):
        """Get MFCC feature from audio record.
            
        `MFCC paper <https://arxiv.org/pdf/1003.4083.pdf>`_
        
        :param record: Record object to get feature from.
        :type record: object
        :param sample_rate: Sample rate for audio record.
        :type sample_rate: int

        :return: MFCC feature vector
        """
        return mfcc(record, fs=sample_rate)
    
    def get_pncc(self, record, sample_rate):
        """Get PNCC feature from audio record.
            
        `PNCC paper <http://www.cs.cmu.edu/~robust/Papers/OnlinePNCC_V25.pdf>`_
        
        :param record: Record object to get feature from.
        :type record: object
        :param sample_rate: Sample rate for audio record.
        :type sample_rate: int

        :return: PNCC feature vector
        """
        return pncc(record, fs=sample_rate)
    
    def get_all_features(self, record, sample_rate=16000, normalize_dim=False):
        """Get all list of features from audio record.
        
        :param record: Record object to get feature from.
        :type record: object
        :param sample_rate: Sample rate for audio record.
        :type sample_rate: int
        :param normalize_dim: Normilize vectors or not.
        :type normalize_dim: bool

        :return: Dictionary of all features
        """
        all_features = {
            'lfcc': self.get_lfcc(record, sample_rate),
            'mfcc': self.get_mfcc(record, sample_rate),
            'pncc': self.get_pncc(record, sample_rate),
            'd_vector': self.get_d_vector(record, sample_rate)
        }
        
        if normalize_dim:
            for feature_name in all_features.keys():
                all_features[feature_name] = self.norm_dim(all_features[feature_name]) if self.is_in_norm_list(feature_name) else all_features[feature_name]
        
        return all_features

    def get_all_features_limited(self, record, sample_rate=16000):
        """Get all list of features from audio record and normalize by 
        default dimension and lenght.
        
        :param record: Record object to get feature from.
        :type record: object
        :param sample_rate: Sample rate for audio record.
        :type sample_rate: int

        :return: Dictionary of all features
        """
        all_features = self.get_all_features(record, sample_rate, normalize_dim=True)
        for feature_name in all_features:
            if feature_name in self.names_to_norm:
                all_features[feature_name] = all_features[feature_name][:self.feature_len]
        return all_features

    def get_all_features_mean_limited(self, record, sample_rate=16000, chunk_len=5):
        chunks_amount = math.floor(len(record)/sample_rate/chunk_len)
        features_sets = {
            'd_vector': [],
            'mfcc': [],
            'lfcc': [],
            'pncc': []
        }
        for i in range(chunks_amount):
            record_chunk = record[i*sample_rate*chunk_len:(i+1)*sample_rate*chunk_len]
            chunk_all_features = self.get_all_features(record_chunk, sample_rate, normalize_dim=True)
            for feature_name in chunk_all_features:
                features_sets[feature_name].append(chunk_all_features[feature_name])
        all_features = {
            'd_vector': None,
            'mfcc': None,
            'lfcc': None,
            'pncc': None
        }
        for feature_name in all_features:
            all_features[feature_name] = np.mean(features_sets[feature_name], axis=0)
        all_features = self.get_all_features(record, sample_rate, normalize_dim=True)
        for feature_name in all_features:
            if feature_name in self.names_to_norm:
                all_features[feature_name] = all_features[feature_name][:self.feature_len]
        return all_features
    
    def is_in_norm_list(self, feature_name):
        """Check feature name in default normaliation list.
        
        :param feature_name: Name of the feature.
        :type feature_name: str

        :return: True or False for feature name in default normalize list.
        """
        return any([name == feature_name for name in self.names_to_norm])
    
    def visualize_spectrogram(self, record, sample_rate):
        """Build plot of spectrogram for audio record.
        
        :param record: Record object.
        :type record: object
        :param sample_rate: Sample rate for audio record.
        :type sample_rate: int
        """
        vis.spectogram(record, sample_rate)
        
    def visualize_features(self, 
                           features, 
                           feature_index='feature', 
                           frame_index='frame index', 
                           normalize_dim=False):
        """Build plot of features for audio record.
        
        :param features: Feature vector.
        :type features: list
        :param feature_index: Feature name index.
        :type feature_index: str
        :param frame_index: Frame name index.
        :type frame_index: str
        :param normalize_dim: Normilize vectors or not.
        :type normalize_dim: bool
        """
        if normalize_dim:
            vis.plot(features, feature_index, frame_index)
        else:
            vis.visualize_features(features, feature_index, frame_index)

    def cosine_similarity(self, x, y):
        """Compare two feature lists by cosine distance.
        
        :param x: First feature vector.
        :type x: list 
        :param y: Second feature vector.
        :type y: list

        :return: Float number from 0 to 1 which shows similarity between two vectors.
        """
        return 1 - spatial.distance.cosine(x, y)

    def compare_two_features_sets(self, features_1, features_2):
        """Compare two feature sets.
        
        :param features_1: First feature set.
        :type features_1: dict
        :param features_2: Second feature set.
        :type features_2: dict

        :return: Float number which shows similarity between two feature sets.
        """
        d_sim = self.cosine_similarity(features_1['d_vector'], features_2['d_vector'])
        lfcc_sim = self.cosine_similarity(features_1['lfcc'], features_2['lfcc'])
        pncc_sim = self.cosine_similarity(features_1['pncc'], features_2['pncc'])
        mfcc_sim = self.cosine_similarity(features_1['mfcc'], features_2['mfcc'])

        sum_sim = d_sim*self.d_weight + lfcc_sim*self.lfcc_weight + pncc_sim*self.pncc_weight + mfcc_sim*self.mfcc_weight
        # TODO: scale sim based on weights
        return sum_sim

    def features_to_json_serializable(self, all_features):
        """Format feature set to json-serializable.
        
        :param all_features: Feature set.
        :type all_features: dict

        :return: Json-serializable feature set dictionary.
        """
        for feature_name in all_features:
            all_features[feature_name] = all_features[feature_name].tolist()
        return all_features
        