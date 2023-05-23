import random
import json
import pickle as pkl
import copy
import math
import numpy as np
import os


def compare(gt_candidates: list, data_candidates: list) -> float:
    """Compare one query gt and prediction and get ap.

    Args:
        gt_candidates(Dict): gt dict for each query,
        data_candidates(Dict): prediction dict for each query
    
    Return average_precision, the label for the first query
    """

    gt_len = len(gt_candidates)
    pred_len = len(data_candidates)
    min_len = min(gt_len, pred_len)

    total_acc = 0
    for idx in range(min_len):
        if gt_candidates[idx] == data_candidates[idx]:
            total_acc += 1
    
    #ap = total_acc/gt_len
    return total_acc, gt_len

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")
    """
    Evaluates the submission for a particular challenge phase adn returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            "status": u"running",
            "when_made_public": None,
            "participant_team": 5,
            "input_file": "https://abc.xyz/path/to/submission/file.json",
            "execution_time": u"123",
            "publication_url": u"ABC",
            "challenge_phase": 1,
            "created_by": u"ABC",
            "stdout_file": "https://abc.xyz/path/to/stdout/file.json",
            "method_name": u"Test",
            "stderr_file": "https://abc.xyz/path/to/stderr/file.json",
            "participant_team_name": u"Test Team",
            "project_url": u"http://foo.bar",
            "method_description": u"ABC",
            "is_public": False,
            "submission_result_file": "https://abc.xyz/path/result/file.json",
            "id": 123,
            "submitted_at": u"2017-03-20T19:22:03.880652Z",
        }
    """
    print(kwargs["submission_metadata"])
    output = {}
    with open(test_annotation_file, 'r') as f:
        gt_dict = json.load(f)
    #gt_dict = load_jsonl(test_annotation_file)
    if phase_codename == "dev" or phase_codename == "eval":
        if phase_codename == "eval":
            phase_name = "Evaluation Phase"
            split_name = "eval_split"
        elif phase_codename == "dev":
            phase_name = "Dev Phase"
            split_name = "val_split"

        #data_dict = load_jsonl(user_submission_file)
        with open(user_submission_file, 'r') as f:
            data_dict = json.load(f)
        print(f"Evaluating for {phase_name} Phase")
        # get AP for each query
        total_acc = 0
        total_gt = 0
        #for idx in gt_dict:
        for key, value in gt_dict.items():
            if key not in data_dict:
                total_acc += 0
                total_gt += len(gt_dict[key])
            else:
                acc, gt_len = compare(gt_dict[key], data_dict[key])
                total_acc += acc
                total_gt += gt_len

        accuracy = total_acc/total_gt

        output["result"] = [
            {
                split_name: {
                    "accuracy": accuracy
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0][split_name]
        #print(output["submission_result"])
        print(f"Completed evaluation for {phase_name} Phase")

    return output
