#!/usr/bin/env python3
from tira.rest_api_client import Client
from pathlib import Path
import json
import click

tira = Client()

def get_index_dir(dataset_id):
    if dataset_id == 'msmarco-passage':
        return tira.get_run_output('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', 'msmarco-passage-trec-dl-2019-judged-20230107-training')
    elif dataset_id == 'beir/webis-touche2020':
        return tira.get_run_output('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', 'argsme-touche-2020-task-1-20230209-training')
    elif dataset_id == 'msmarco-subsample':
        return str(Path('../../data/ms-marco-subsample').absolute().resolve())
    else:
        raise ValueError('foo')

def partitioned_qpptk_queries(partition):
    # reformat so that they are compatible with the tirex format
    target_file = Path(f'../../data/qpptk_partitioned/partition-{partition}/queries.jsonl')
    if target_file.exists():
        return target_file.parent

    queries = set()
    with open('../../data/queries.jsonl', 'r') as f:
        for l in f:
            l = json.loads(l)
            queries.add(l['query'])
    ret = []
    qid = 1

    for query in queries:
        signs_to_remove = [':', '#', '-', '?', '"', '.', '\\', '[', ']', '(', ')', '|']
        skip = False
        for s in signs_to_remove:
            if s in query:
                skip = True

        if skip or not query.isascii() or query.isdigit():
            continue

        ret.append(json.dumps({"qid": str(qid), "query": query}) + '\n')
        qid += 1
    
    print(len(ret))
    chunks = [ret[100*i:100*(i+1)] for i in range(int(len(ret)/100) + 1)]
    s = set()
    for chunk_num, chunk in zip(range(len(chunks)), chunks):
        print(chunk_num, '->', len(chunk))
        target_file = Path(f'../../data/qpptk_partitioned/partition-{chunk_num}/queries.jsonl')
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, 'w') as f:
            for c in chunk:
                s.add(c)
                f.write(c)

    print(len(s))

    return str(target_file.parent)


def run_qpptk(dataset_id, partition):
    input_dir = partitioned_qpptk_queries(partition)
    output_dir = (Path(input_dir) / dataset_id.replace('/', '-')).absolute().resolve()

    if output_dir.exists():
        return

    index_dir = get_index_dir(dataset_id)
    system_details = tira.public_system_details('qpptk', 'all-predictors')
    tira.local_execution.run(
        image=system_details['public_image_name'],
        command=system_details['command'],
        output_dir=output_dir,
        input_dir=input_dir,
        input_run=index_dir,
    )


@click.command()
@click.option('--dataset', required=True, type=click.Choice(['msmarco-passage', 'beir/webis-touche2020', 'msmarco-subsample']))
@click.option('--partition', required=True, type=int)
@click.option('--num', default=10, type=int)
def main(dataset, partition, num):
    to_predict = list(range(partition, partition + num))
    print(f'Run {dataset} preds on partitions {to_predict}')

    for p in to_predict:
        run_qpptk(dataset, p)

if __name__ == main():
    main()

