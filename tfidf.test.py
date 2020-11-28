from tfidf import TfIdf
import json

docs = {
    'd1': {
        'keywords': {
            'a': {
                'f': 10,
                'w': {
                    'TITLE': 2.0
                }
            }
        }
    }
}

ti = TfIdf(docs)

for kd, d in docs.items():
    for kw, t in d['keywords'].items():
        docs[kd]['keywords'][kw]['tf_idf'] = ti.tf_idf(kw, kd)

print(json.dumps(docs, sort_keys=False, indent=4))