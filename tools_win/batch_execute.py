import os
import re
import sys
import shutil
import subprocess
from multiprocessing import Pool
from datetime import datetime
from typing import List
import yaml

timestamp = datetime.now()

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TOOLS_DIR)

INPUT_DIR = os.path.join(TOOLS_DIR, 'in')
OUTPUT_DIR = os.path.join(TOOLS_DIR, 'out')
ERROR_DIR = os.path.join(TOOLS_DIR, 'err')

SOLVER_DIR = os.path.join(ROOT_DIR, 'vs', 'solver')
SOURCE_FILE = os.path.join(SOLVER_DIR, 'src', 'solver.cpp')

EXEC_DIR = os.path.join(SOLVER_DIR, 'bin', 'Release')
EXEC_BIN = os.path.join(EXEC_DIR, 'solver.exe')
TESTER_BIN = os.path.join(TOOLS_DIR, 'tester.exe')



def store_values_from_stderr(result: dict, key_list: List[str], stderr_file: str):
    with open(stderr_file, 'r', encoding='utf-8') as f:
        lines = str(f.read()).split('\n')
    for line in lines:
        for key in key_list:
            pattern = fr'^{key} = (\d+)'
            m = re.match(pattern, line)
            if m:
                result[key] = int(m.group(1))

def run_wrapper(cmd: str):
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':

    tag = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    if len(sys.argv) >= 2:
        tag += '_' + sys.argv[1]
    print(tag)

    assert not os.path.exists(os.path.join(TOOLS_DIR, 'submissions', tag))

    shutil.rmtree(OUTPUT_DIR)
    shutil.rmtree(ERROR_DIR)

    os.makedirs(OUTPUT_DIR)
    os.makedirs(ERROR_DIR)

    cmds = []
    for seed in range(0, 100):
        input_file = os.path.join(INPUT_DIR, f'{seed:04d}.txt')
        output_file = os.path.join(OUTPUT_DIR, f'{seed:04d}.txt')
        error_file = os.path.join(ERROR_DIR, f'{seed:04d}.txt')
        cmd = f'{TESTER_BIN} {EXEC_BIN} < {input_file} > {output_file} 2> {error_file}'
        cmds.append(cmd)

    pool = Pool(8)
    pool.map(run_wrapper, cmds)

    results = []
    key_list = ['Score', 'Number of wrong answers', 'Placement cost', 'Measurement cost', 'Measurement count']
    for seed in range(0, 100):
        error_file = os.path.join(ERROR_DIR, f'{seed:04d}.txt')
        result = dict()
        result['Seed'] = seed
        result['Score'] = -1
        store_values_from_stderr(result, key_list, error_file)
        results.append(result)
    
    submissions_dir = os.path.join('submissions', tag)
    os.makedirs(submissions_dir)
    shutil.copytree(OUTPUT_DIR, os.path.join(submissions_dir, 'out'))
    shutil.copytree(ERROR_DIR, os.path.join(submissions_dir, 'err'))
    shutil.copy2(SOURCE_FILE, submissions_dir)
    with open(os.path.join(submissions_dir, 'results.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(results, f, sort_keys=False)