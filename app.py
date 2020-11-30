from flask import Flask, request, jsonify

from featurizer import AudioFeaturizer
from dtln import DTLNproc


app = Flask(__name__)
audio_featurizer = AudioFeaturizer()
dtln_proc = DTLNproc()


@app.route('/features', methods=['GET'])
def features():
    url = request.args.get('url')
    dtln_on = request.args.get('dtln', default=True)
    record, sample_rate = audio_featurizer.read_file_by_url(url)
    if dtln_on:
        record = dtln_proc.process_record(record, sample_rate)
    all_features = audio_featurizer.get_all_features_limited(record, sample_rate)
    all_features = audio_featurizer.features_to_json_serializable(all_features)
    result = dict(features=all_features)
    return jsonify(result)


@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    res_sim = audio_featurizer.compare_two_features_sets(data['features_1'], data['features_2'])
    result = dict(similarity=res_sim)
    return jsonify(result)
