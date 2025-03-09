#!/usr/bin/env python3
from tira.rest_api_client import Client
from pathlib import Path
from shutil import copytree

tira = Client(allow_local_execution=True)

output_dir = Path(tira.get_run_output('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', '../../data/ms-marco-subsample')) / 'index'

assert output_dir.exists()
copytree(output_dir, '../../data/ms-marco-subsample/index')

