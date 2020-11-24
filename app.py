from flask import Flask, request, jsonify

from featurizer import AudioFeaturizer


app = Flask(__name__)
audio_featurizer = AudioFeaturizer()


@app.route('/features', methods=['GET'])
def features():
    url = request.args.get('url')
    record, sample_rate = audio_featurizer.read_file_by_url(url)
    all_features = audio_featurizer.get_all_features_limited(record, sample_rate)
    result = dict(features=all_features)
    return jsonify(result)


@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    res_sim = audio_featurizer(data['features_1'], data['features_2'])
    result = dict(similarity=res_sim)
    return jsonify(result)
