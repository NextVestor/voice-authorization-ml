from flask import Flask, request, jsonify

from featurizer import AudioFeaturizer
from dtln import DTLNproc


app = Flask(__name__)
audio_featurizer = AudioFeaturizer()
dtln_proc = DTLNproc()


@app.route('/features', methods=['GET'])
def features():
    """Get features for voice record.

    .. :quickref: Get features; Get features for record.

    **Example request**:

    .. sourcecode:: http

        GET /features?url=https://example.com/record.wav HTTP/1.1
        Host: example.com
        Accept: application/json

    **Example response**:

    .. sourcecode:: http

        HTTP/1.1 200 OK
        Vary: Accept
        Content-Type: application/json

        {
            'features': {
                'd_vector': [0.01,0.01,0.0,0.0,0.01,0.14],
                'mfcc': [-0.22,-0.12,-0.39,-0.51,-0.37],
                'pncc': [0.04,0.04,0.15,0.18,0.14],
                'lfcc': [-0.85,-0.76,-0.69,-0.55,-0.35]
            }
        }

    :query url: url path to the wav file
    :query dtln: True/False; use speech improvement algorithm or not
    :<json dict features: feature set for second
    :<json array d_vector: d-vector for voice record
    :<json array mfcc: mfcc feature vector for voice record
    :<json array pfcc: pfcc feature vector for voice record
    :<json array lfcc: lfcc feature vector for voice record
    :resheader Content-Type: application/json
    :status 200: features extracted
    """
    url = request.args.get('url')
    dtln_on = request.args.get('dtln', default=True)
    mean_on = request.args.get('mean', default=False)
    record, sample_rate = audio_featurizer.read_file_by_url(url)
    if dtln_on:
        record = dtln_proc.process_record(record, sample_rate)
    if mean_on:
        all_features = audio_featurizer.get_all_features_mean_limited(record, sample_rate)
    else:
        # TODO: change logic (mean as parameter?)
        all_features = audio_featurizer.get_all_features_limited(record, sample_rate)
    all_features = audio_featurizer.features_to_json_serializable(all_features)
    result = dict(features=all_features)
    return jsonify(result)


@app.route('/compare', methods=['POST'])
def compare():
    """Compare two sets of features. Return comparison distance \
    between two sets of records.

    .. :quickref: Compare features; Return comparison distance between\
    two sets of features.

    **Example request**:

    .. sourcecode:: http

        POST /compare HTTP/1.1
        Host: example.com
        Accept: application/json

        {
            'features1': {
                'd_vector': [0.01,0.01,0.0,0.0,0.01,0.14],
                'mfcc': [-0.22,-0.12,-0.39,-0.51,-0.37],
                'pncc': [0.04,0.04,0.15,0.18,0.14],
                'lfcc': [-0.85,-0.76,-0.69,-0.55,-0.35]
            },
            'features2': {
                'd_vector': [0.02,0.03,0.1,0.1,0.015,0.14],
                'mfcc': [-0.42,-0.22,-0.4,-0.67,-0.3],
                'pncc': [0.05,0.01,0.25,0.1,0.35],
                'lfcc': [-0.79,-0.9,-0.7,-0.67,-0.45]
            }
        }

    **Example response**:

    .. sourcecode:: http

        HTTP/1.1 200 OK
        Vary: Accept
        Content-Type: application/json

        {
            'similarity': 0.7
        }

    :<json dict features1: feature set for first record
    :<json dict features2: feature set for second record
    :<json array d_vector: d-vector for voice record
    :<json array mfcc: mfcc feature vector for voice record
    :<json array pfcc: pfcc feature vector for voice record
    :<json array lfcc: lfcc feature vector for voice record
    :>json float similarity: comparison distance between two feature sets
    :resheader Content-Type: application/json
    :status 200: features compared
    """
    data = request.get_json()
    res_sim = audio_featurizer.compare_two_features_sets(data['features_1'], data['features_2'])
    result = dict(similarity=res_sim)
    return jsonify(result)
