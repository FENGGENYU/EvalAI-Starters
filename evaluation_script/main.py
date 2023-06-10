import random
import json
import pickle as pkl
import copy
import math
import numpy as np
import os


def calc_mIoU(labels,preds):
	ious = []
	
	inter = 0
	union = 0
	
	for pred, label in zip(preds, labels):
		
		ninds = list(set(pred.tolist() + label.tolist()))

		for ind in ninds:
			pn=np.zeros(pred.shape, dtype=int)
			ln=np.zeros(label.shape, dtype=int)

			pind=(pred ==ind).nonzero()[0]
			lind=(label == ind).nonzero()[0]
			pn[pind]=1
			ln[lind]=1
			inter += (pn & ln).sum().item()
			union += (pn | ln).sum().item()

			if union == 0:
				continue
			iou=(1.*inter)/(1.*union)
			ious.append(iou)

	miou = np.array(ious).mean()
	return miou

def calc_seg_miou(pred_label_list,gt_labels_list, segment_list):
	A_labels = []
	A_preds = []
	# for samp_segments,samp_labels,seg_preds in zip(
	#data['samp_segments'],gt_labels,A_seg_preds ##
	#must start from index 0
	for i in range(len(pred_label_list)):
		samp_segments = segment_list[i]
		seg_preds = pred_label_list[i]
		gt_labels = gt_labels_list[i]
		samp_preds = np.zeros(samp_segments.shape[0], dtype=int)-1 
		samp_labels = np.zeros(samp_segments.shape[0], dtype=int)-1 
	
		for j, (p,g) in enumerate(zip(seg_preds,gt_labels)):
			inds= np.nonzero((samp_segments == j))[0].flatten()
			samp_preds[inds]=p
			samp_labels[inds]=g

		assert(samp_preds>=0).all(), 'some label left'
		A_labels.append(samp_labels)
		A_preds.append(samp_preds)

	iou = calc_mIoU(A_labels,A_preds)
	return iou

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
		
		test_annotation_file_seg = test_annotation_file[:-8] +'.npz'
		seg_data = np.load(test_annotation_file_seg)
		seg_name_list = seg_data['names'].tolist()
		segments_array = seg_data['segments']

		print(f"Evaluating for {phase_name} Phase")
		# get AP for each query
		total_acc = 0
		total_gt = 0
		test_label_list = []
		gt_label_list = []
		segment_list = []
		#for idx in gt_dict:
		for key, value in gt_dict.items():
			if key not in data_dict:
				total_acc += 0
				total_gt += len(gt_dict[key])

				test_label_list.append([0 for _ in range(len(gt_dict[key]))])
				gt_label_list.append(gt_dict[key])
				index = seg_name_list.index(key)
				segment_list.append(segments_array[index, :])

			else:
				acc, gt_len = compare(gt_dict[key], data_dict[key])
				total_acc += acc
				total_gt += gt_len
				test_label_list.append(data_dict[key])
				gt_label_list.append(gt_dict[key])

				index = seg_name_list.index(key)
				segment_list.append(segments_array[index, :])

		accuracy = total_acc/total_gt
		iou = calc_seg_miou(test_label_list, gt_label_list, segment_list)

		output["result"] = [
			{
				split_name: {
					"accuracy": accuracy,
					"iou": iou,
				}
			}
		]
		# To display the results in the result file
		output["submission_result"] = output["result"][0][split_name]
		print(output["submission_result"])
		print(f"Completed evaluation for {phase_name} Phase")

	return output
