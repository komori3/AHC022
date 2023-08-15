import os
import yaml
import math
from collections import defaultdict

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



def show_standings(dict_submission_to_total_score):
    max_length = len('submission')
    list_total_score_to_submission = []
    for submission, total_score in dict_submission_to_total_score.items():
        max_length = max(max_length, len(submission))
        list_total_score_to_submission.append((total_score, submission))

    list_total_score_to_submission.sort()

    space = max_length - len('submission') + 4
    print('submission' + (' ' * space) + 'score')
    print('-' * 50)
    for total_score, submission in list_total_score_to_submission:
        space = max_length - len(submission) + 4
        print(submission + (' ' * space) + str(total_score))

if __name__ == "__main__":

    submissions_dir = os.path.join('submissions')
    submission_to_results = {}
    for tag in os.listdir(submissions_dir):
        results_file = os.path.join(submissions_dir, tag, 'results.yaml')
        with open(results_file) as f:
            submission_to_results[tag] = yaml.safe_load(f)

    # seed_best = defaultdict(lambda: -1)
    # for tag, results in submission_to_results.items():
    #     for result in results:
    #         seed_best[result['Seed']] = max(seed_best[result['Seed']], result['Score'])

    dict_submission_to_total_score = defaultdict(lambda: 0.0)
    for tag, results in submission_to_results.items():
        ctr = 0
        for result in results:
            ctr += 1
            if result['Score'] == -1: continue
            # dict_submission_to_total_score[tag] += result['Score'] / seed_best[result['Seed']]
            dict_submission_to_total_score[tag] += math.log(result['Score'])
        dict_submission_to_total_score[tag] /= ctr

    show_standings(dict_submission_to_total_score)