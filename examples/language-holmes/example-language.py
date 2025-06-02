import json
import os
import random as rd
from collections import Counter
from multiprocessing import Pool
from random import random, shuffle

from tqdm import tqdm
from utility import load_texts, story2instances

from cobweb.cobweb import CobwebTree
from cobweb.visualize import visualize

if __name__ == "__main__":


	# Configuration:
	json_name = "holmes_stories.json"
	verbose = True
	limit = None
	window = 10  # the size of the "gram"
	least_frequency = 100  # used to filter out the words having frequency less than some specified one.
	seed = 123  # random seed for shuffling instances


	""" Load and preprocess the text data used """
	if verbose:
		print("Start the loading and preprocessing process.")

	# Create the integrated text from several text file (stories):
	if not os.path.isfile(json_name):
		if verbose:
			print("\nReading and preprocessing...")
		stories_dir = "./data"
		stories = list(load_texts(stories_dir, limit=limit))
		with open(json_name, "w") as fout:
			json.dump(stories, fout, indent=4)
	else:
		if verbose:
			print("\nLoading the preprocessed stories...")
		with open(json_name) as fin:
			stories = json.load(fin)


	# A 200-first-word preview of some preprocessed story:
	print("\nPreview of the 200 first words of the first preprocessed story:")
	print(stories[0][:200])


	# You may see the overall word frequencies (50 most frequent words):
	print("\npreview of the 100 most frequent words frequency:")
	overall_freq = Counter([word for story in stories for word in story])
	print(overall_freq.most_common(100))



	""" Generate the instances for Cobweb learning, the 'n-grams' """
	# Filter out the words having frequency >= least_frequency only:
	stories = [[word for word in story if overall_freq[word] >= least_frequency] for story in stories]


	# Generate instances (along with their story and anchor indices):
	instances = []
	with Pool() as pool:
		processed_stories = pool.starmap(story2instances, [(story, window) for story in stories])
		for story_idx, story_instances in enumerate(processed_stories):
			for anchor_idx, instance in story_instances:
				instances.append((instance, story_idx, anchor_idx))

	rd.seed(seed)
	shuffle(instances)

	# print(instances[17])


	""" Naive train and test processes """

	# # Generate inststances (only, instead of instance + story_id + anchor_id)
	# instances = []
	# with Pool() as pool:
	# 	processed_stories = pool.starmap(story2instances, [(story, window) for story in stories])
	# 	for story_idx, story_instances in enumerate(processed_stories):
	# 		for anchor_idx, instance in story_instances:
	# 			instances.append(instance)
	# rd.seed(seed)
	# shuffle(instances)

	# # Train the Cobweb tree:
	# tree = CobwebTree(0.000001, False, 0, True, False)
	# for instance in instances:
	# 	tree.ifit(instance)

	# # Visualize the concepts
	# visualize(tree)

	# # Test the Cobweb tree:
	# instances_no_anchor()








	""" Incremental train-test trials of Cobweb.
		To see the evolution of performance (or predicted probability of the anchor word) vs. # of training instances,
		In each trial, Cobweb first test the upcoming instance then learn it.
	"""
	if verbose:
		print("\nStart training and evaluating.")
	n_training_words = 0
	occurances = Counter()
	training_queue = []
	outfile = f"cobweb_{window}_holmes_out"
	tree = CobwebTree(0.000001, False, 0, True, False)
	with open(outfile + ".csv", 'w') as fout:
			fout.write("n_training_words,n_training_stories,model,word,word_freq,word_obs_count,vocab_size,pred_word,prob_word,correct,story\n")

	for idx, (instance, story_idx, anchor_idx) in enumerate(tqdm(instances)):
		anchor = list(instance['anchor'].keys())[0]

		# Story as a sentence completion form
		story = stories[story_idx]
		text = " ".join([w for w in story[max(0, anchor_idx-window):anchor_idx]])
		text += " _ "
		text += " ".join([w for w in story[max(0, anchor_idx+1):anchor_idx+window+1]])

		instance_no_anchor = {'context': instance['context']}
		probs_pred = tree.predict_probs(instance_no_anchor, 100, False, False, 1)
		prob_pred = 0
		anchor_pred = "NONE"
		if "anchor" in probs_pred and anchor in probs_pred['anchor']:
			prob_pred = probs_pred['anchor'][anchor]
		if "anchor" in probs_pred and len(probs_pred["anchor"]) > 0:
			anchor_pred = sorted([(probs_pred["anchor"][word], random(), word) for word in probs_pred['anchor']], reverse=True)[0][2]

		training_queue.append(instance)
		with open(outfile + ".csv", "a") as fout:
			fout.write("{},{},cobweb,{},{},{},{},{},{},{},{}\n".format(
					n_training_words,  # n_training_words
					story_idx,  # n_training_stories
					anchor,  # word
					overall_freq[anchor],  # word_freq
					occurances[anchor],  # word_obs_count
					len(occurances),  # vocab_size
					anchor_pred,  # pred_word
					prob_pred,  # prob_word
					1 if anchor_pred == anchor else 0,  # correct
					text  # story
					))

		if len(training_queue) > 0:
				old_inst = training_queue.pop(0)
				tree.ifit(old_inst)
				old_anchor = list(old_inst['anchor'].keys())[0]
				occurances[old_anchor] += 1
				n_training_words += 1


	visualize(tree)












