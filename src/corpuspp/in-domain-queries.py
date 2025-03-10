#!/usr/bin/env python3
import json

CORPORA = [
    '/mnt/ceph/tira/data/datasets/training-datasets/ir-benchmarks/nfcorpus-test-20230107-training/queries.jsonl',
    '/mnt/ceph/tira/data/datasets/training-datasets/ir-benchmarks/msmarco-passage-trec-dl-2019-judged-20230107-training/queries.jsonl',
    '/mnt/ceph/tira/data/datasets/training-datasets/ir-benchmarks/argsme-touche-2020-task-1-20230209-training/queries.jsonl',
]
ret = []
for c in CORPORA:
    cnt = 0
    with open(c, 'r') as f:
        if 'nfcorpus' in c:
            suffix = 'nfcorpus'
        elif 'msmarco' in c:
            suffix = 'msmarco'
        elif 'argsme' in c:
            suffix = 'argsme'
        else:
            raise ValueError('foo')
        for l in f:
            cnt += 1
            if cnt > 51:
                continue
            l = json.loads(l)
            l['qid'] = l['qid'] + '-' + suffix
            ret.append(json.dumps(l) + '\n')

with open('../../data/qpptk_partitioned/partition-999/queries.jsonl', 'w') as f:
    for l in ret:
        f.write(l)

print(len(ret))
