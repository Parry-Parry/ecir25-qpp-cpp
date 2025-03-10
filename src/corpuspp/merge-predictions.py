#!/usr/bin/env python3
from glob import glob
import json
import gzip
from tqdm import tqdm

DATASETS = [
    'beir-webis-touche2020',
    'msmarco-passage',
    'msmarco-subsample',
    'nfcorpus',
    'cranfield',
]

def matches(dataset):
    ret = {}
    for file in glob(f'../../data/qpptk_partitioned/partition-*/{dataset}/queries.jsonl'):
        with open(file, 'r') as f:
            for l in f:
                l = json.loads(l)
                assert l['qid'] not in ret
                ret[l['qid']] = l

    print(dataset, len(ret))
    return ret

qid_to_dataset = {}
qid_to_query = {}
query_to_qid = {}

for file in tqdm(glob(f'../../data/qpptk_partitioned/partition-*/queries.jsonl'), 'partitions'):
   with open(file, 'r') as f:
        for l in f:
            l = json.loads(l)
            qid_to_query[l['qid']] = l['query']
            query_to_qid[l['query']] = l['qid']

for d in tqdm(DATASETS, 'predictions'):
    for qid, v in matches(d).items():
        if qid not in qid_to_dataset:
            qid_to_dataset[qid] = {}

        qid_to_dataset[qid][d] = v

preds = []

with open('../../data/queries.jsonl', 'r') as f:
    for l in tqdm(f):
        l = json.loads(l)
        if l['query'] not in query_to_qid:
            continue

        if query_to_qid[l['query']] not in qid_to_dataset or len(qid_to_dataset[query_to_qid[l['query']]]) != len(DATASETS):
            continue

        l['qpptk_predictions'] = qid_to_dataset[query_to_qid[l['query']]]
        preds.append(json.dumps(l) + '\n')
   

print('=>', len(preds))
with gzip.open('../../data/queries-with-predictions.jsonl.gz', 'wt') as f:
    for l in preds:
        f.write(l)

preds = []

for query in qid_to_dataset.keys():
    if '-cranfield' in query:
        source = 'cranfield'
    elif '-argsme' in query:
        source = 'beir-webis-touche2020'
    elif '-nfcorpus' in query:
        source = 'nfcorpus'
    elif '-msmarco' in query:
        source = 'msmarco-passage'
    else:
        continue
    if len(qid_to_dataset[query]) != len(DATASETS):
        continue

    preds.append(json.dumps({'query': query, 'source': source, 'qpptk_predictions': qid_to_dataset[query]}) + '\n')

print('=>', len(preds))
with gzip.open('../../data/in-domain-queries-with-predictions.jsonl.gz', 'wt') as f:
    for l in preds:
        f.write(l)

