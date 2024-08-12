from cobweb.cobweb import CobwebTree
from cobweb.visualize import visualize
from random import shuffle, seed, sample
import time
import csv
import numpy as np
import pandas as pd
from copy import copy, deepcopy
from tqdm import tqdm

"""
The example dataset is available at: 
https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset?select=dataset.csv
4921 cases in total.
"""

# Configurations:
size_tr = 100  # the size of the training set (so the rest data is used for prediction)
random_seed = 32
verbose = True


# severity = {}
# with open("symptom-severity.csv", 'r') as fin:
# 	csv_reader = csv.reader(fin)
# 	header = next(csv_reader)
# 	for row in csv_reader:
# 		severity[row[0].lower().replace(' ', '')] = int(row[1])
# # print(severity)


""" Data preprocessing. Table -> Instances (dictionaries) """
instances = []
with open("/home/zwang910/Research/packages/cobweb/examples/tabular/diagnose.csv", 'r') as fin:
	csv_reader = csv.reader(fin)
	header = next(csv_reader)
	for row in csv_reader:
		instance = {'Disease': {row[0].lower().replace(' ', '-'): 1}}
		symptom_dict = {}
		for i in range(1, len(row)):
			if row[i] == '':
				break
			symptom_dict[row[i].lower().replace(' ', '')] = 1
		instance['symptom'] = symptom_dict
		# severity_ls = [severity[symptom] for symptom in list(symptom_dict.keys())]
		# instance['severity'] = {str(sum(severity_ls)): 1}
		instances.append(instance)

# Shuffle the learned instances:
seed(random_seed)
shuffle(instances)
instances_tr = instances[:size_tr]
instances_te = instances[size_tr:]
diseases_te = [list(instance['Disease'].keys())[0] for instance in instances_te]
instances_te = [{k: v for k, v in instance.items() if k != 'Disease'} for instance in instances_te]

# print(instances_tr[13])

""" Train Cobweb """
if verbose:
	print(f"Start training. Train with {size_tr} samples.")
tree = CobwebTree(0.001, False, 0, True, False)
for instance in tqdm(instances_tr):
	tree.ifit(instance)
visualize(tree)  # a visualization of the trained Cobweb tree

# given some test case:
# instance = instances_te[0]
# probs_pred = tree.predict_probs(instance, 50, False, False, 1)
# disease_pred = sorted([(prob, disease) for (disease, prob) in probs_pred['disease'].items()], reverse=True)[0][1]

# print(probs_pred)
# print(disease_pred)
# print(diseases_te[0])

""" Evaluation: Cobweb making predictions on the rest data """
if verbose:
	print(f"\nStart testing. Test with {len(instances_te)} samples.")
n_correct = 0
diseases_pred = []
for i in tqdm(range(len(instances_te))):
	instance = instances_te[i]
	probs_pred = tree.predict_probs(instance, 50, False, False)
	# probs_pred = tree.categorize(instance).predict_probs()
	disease_pred = sorted([(prob, disease) for (disease, prob) in probs_pred['Disease'].items()], reverse=True)[0][1]
	if disease_pred == diseases_te[i]:
		n_correct += 1
	# if i <= 20:
	# 	print("\n")
	# 	print(diseases_te[i], disease_pred)
	# 	print(probs_pred['disease'])
	diseases_pred.append(disease_pred)

accuracy = n_correct / len(instances_te)
print(f"The test accuracy of Cobweb after training {size_tr} samples: {accuracy}")


# Predict the symptoms of a disease:
instance_te_disease = {'Disease': {'diabetes-': 1}}
probs_pred = tree.categorize(instance_te_disease).get_basic_level().predict_probs()
# probs_pred = tree.predict_probs(instance, 10, False, False)
symptoms_pred = sorted([(prob, symptom) for (symptom, prob) in probs_pred['symptom'].items()], reverse=True)[:10]
print(symptoms_pred)

instances_te_reverse = {'symptom': {'vomiting': 1, 'abdominal_pain': 1, 'loss_of_appetite': 1, 'fatigue': 1, 'yellowish_skin': 1, 'nausea': 1, 'high_fever': 1, 'dark_urine': 1}}
probs_pred = tree.predict_probs(instance, 50, False, False)
disease_pred = sorted([(prob, disease) for (disease, prob) in probs_pred['Disease'].items()], reverse=True)[0][1]
print(disease_pred)





