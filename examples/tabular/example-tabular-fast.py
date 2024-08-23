from cobweb.cobweb_discrete import CobwebTree
from cobweb.visualize import visualize
from random import shuffle, seed, sample
import time
import csv
import numpy as np
import pandas as pd
from copy import copy, deepcopy
from tqdm import tqdm
import json
import os

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
with open("/home/zwang910/Research/packages/cmkae-cobwbe/cobweb/examples/tabular/diagnose.csv", 'r') as fin:
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

print(instances[0])

# create a mapping from string to integer:
disease_set = set()
symptom_set = set()

for instance in instances:
	disease_set.add(list(instance['Disease'].keys())[0])
	for symptom in instance['symptom']:
		symptom_set.add(symptom)

# if path exists, load the variable_dict from the file:
variable_list = list(symptom_set) + list(disease_set)
if os.path.exists("variable_dict.json"):
	with open("variable_dict.json", 'r') as fin:
		variable_dict = json.load(fin)
	print("Load variable_dict from file.")
else:
	variable_dict = {variable: i for i, variable in enumerate(variable_list)}
	# save the variable_dict to a file:
	with open("variable_dict.json", 'w') as fout:
		json.dump(variable_dict, fout)
# print(variable_dict)

# map back to the instances:
for instance in instances:
	disease = list(instance['Disease'].keys())[0]
	symptom_dict = instance['symptom']
	instance['Disease'] = {variable_dict[disease]: 1}
	instance['symptom'] = {variable_dict[symptom]: 1 for symptom in symptom_dict}

# change the keys 'Disease' to -1, and 'symptom' to -2:
disease_key = len(variable_list) + 1
symptom_key = len(variable_list) + 2
for instance in instances:
	instance[len(variable_list) + 1] = instance.pop('Disease')
	instance[len(variable_list) + 2] = instance.pop('symptom')

# Shuffle the learned instances:
seed(random_seed)
shuffle(instances)
instances_tr = instances[:size_tr]
instances_te = instances[size_tr:]
diseases_te = [list(instance[disease_key].keys())[0] for instance in instances_te]
instances_te = [{k: v for k, v in instance.items() if k != disease_key} for instance in instances_te]

# print(instances_tr[13])

""" Train Cobweb """
if verbose:
	print(f"Start training. Train with {size_tr} samples.")
tree = CobwebTree(0.001, False, 0, True, False)
for instance in tqdm(instances_tr):
	tree.ifit(instance, mode=1)
visualize(tree)  # a visualization of the trained Cobweb tree

# tree_json = tree.dump_json()
# with open("cobweb-tabular.json", 'w') as fout:
# 	fout.write(tree_json)

# write attr_vals to file
# attr_vals = tree.get_attr_vals()
# print(attr_vals)


# save the tree
tree.write_json_stream("cobweb-tabular.json")
# exit()
# load the tree
# tree2 = CobwebTree(0.001, False, 0, True, False)
# print(tree.dump_json()[:300])
tree = CobwebTree(0.001, False, 0, True, False)
print("=====================================")
tree.load_json_stream("cobweb-tabular.json")
# tree2.load_json_stream("cobweb-tabular.json")
# attr_vals2 = tree2.get_attr_vals()
# print(attr_vals2)

# print(tree.dump_json()[:300])
# save the tree
# tree.write_json_stream("cobweb-tabular-check.json")
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
	_, probs_pred = tree.predict_probs(instance, 50, False, False)
	# probs_pred = tree.categorize(instance).predict_probs()
	disease_pred = sorted([(prob, disease) for (disease, prob) in probs_pred[disease_key].items()], reverse=True)[0][1]
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
instance_te_disease = {disease_key: {variable_dict['diabetes-']: 1}}
probs_pred = tree.categorize(instance_te_disease).get_basic_level().predict_probs()
# probs_pred = tree.predict_probs(instance, 10, False, False)
symptoms_pred = sorted([(prob, symptom) for (symptom, prob) in probs_pred[symptom_key].items()], reverse=True)[:10]
print(symptoms_pred)

# instances_te_reverse = {-1: {'vomiting': 1, 'abdominal_pain': 1, 'loss_of_appetite': 1, 'fatigue': 1, 'yellowish_skin': 1, 'nausea': 1, 'high_fever': 1, 'dark_urine': 1}}
instances_te_reverse = {disease_key: {variable_dict['vomiting']: 1, variable_dict['abdominal_pain']: 1, variable_dict['loss_of_appetite']: 1, variable_dict['fatigue']: 1, variable_dict['yellowish_skin']: 1, variable_dict['nausea']: 1, variable_dict['high_fever']: 1, variable_dict['dark_urine']: 1}}
_, probs_pred = tree.predict_probs(instance, 50, False, False)
disease_pred = sorted([(prob, disease) for (disease, prob) in probs_pred[disease_key].items()], reverse=True)[0][1]
print(disease_pred)





