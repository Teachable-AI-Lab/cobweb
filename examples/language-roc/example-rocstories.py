import csv
import random as rd

# import re
import string

import nltk
from nltk.tokenize import word_tokenize

# from multiprocessing import Pool
# from utility import load_texts, story2instances
from cobweb.cobweb import CobwebTree
from cobweb.visualize import visualize

# def get_most_frequent_words(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         text = file.read().lower()  # Read the file and convert text to lowercase

#     # Tokenize the words using regular expression
#     words = re.findall(r'\b\w+\b', text)

#     # Count the frequency of each word
#     word_freq = Counter(words)

#     # Get the 50 most common words
#     most_common_words = word_freq.most_common(100)

#     return most_common_words

# if __name__ == "__main__":
#     file_path = 'ROCStories_winter2017 - ROCStories_winter2017.txt'  # Replace 'your_file.txt' with the path to your text file
#     frequent_words = get_most_frequent_words(file_path)
#     print("100 Most Frequent Words:")
#     for word, frequency in frequent_words:
#         print(f"{word}: {frequency}")

# Most frequent verbs:
# had: 23529
# went: 10865; go: 5140
# got: 10453; get: 5496
# wanted: 7726
# took: 4951
# found: 4689

# json_name = "rocstories.json"
verbose = True
limit = None
window = 5  # the size of the "gram"
least_frequency = 100  # used to filter out the words having frequency less than some specified one.
seed = 123  # random seed for shuffling instances
rd.seed(seed)
# verbs = ['had', 'went', 'got', 'wanted', 'took', 'found']
verbs = ['had'] * 5 + ['went'] * 5 + ['got'] * 5 + ['wanted'] * 5 + ['took'] * 5 + ['found'] * 5


nltk.download('punkt')  # Download NLTK tokenizer data

def load_and_tokenize_text(file_path):
    rows = []
    tokenized_rows = []
    with open(file_path, encoding='utf-8') as file:
        for line in file:
            # Tokenize the words in each line
            tokenized_words = word_tokenize(line.lower().strip())
            tokenized_words = [word for word in tokenized_words if word not in string.punctuation]
            rows.append(line.strip())
            tokenized_rows.append(tokenized_words)
    return rows, tokenized_rows


def sentence2instance(tokens, window, verb):
	instance = {}
	anchor_id = 0
	# Find the anchor:
	for i in range(len(tokens)):
		if tokens[i] == verb:
			instance['anchor'] = {tokens[i]: 1}
			anchor_id = i
			break
	before_anchor = tokens[max(0, anchor_id - window):anchor_id]
	after_anchor = tokens[anchor_id + 1:min(len(tokens), anchor_id + 1 + window)]
	context_before = {}
	context_after = {}
	for i, w in enumerate(before_anchor):
		context_before[w] = 1 / abs(len(before_anchor) - i)
	for i, w in enumerate(after_anchor):
		context_after[w] = 1 / (i + 1)
	instance['context-before'] = context_before
	instance['context-after'] = context_after
	return instance


def file2instance(file_path, window, shuffle=False):
	rows, token_rows = load_and_tokenize_text(file_path)
	if shuffle:
		ids = list(range(len(rows)))
		# print(ids)
		rd.shuffle(ids)
		rows = create_list_in_order(rows, ids)
		token_rows = create_list_in_order(token_rows, ids)
		verbs_tr = create_list_in_order(verbs, ids)
	else:
		verbs_tr = verbs
	return [sentence2instance(token_rows[i], window, verbs_tr[i]) for i in range(len(token_rows))], rows


def create_list_in_order(original_list, index_order):
	new_list = [original_list[i] for i in index_order]
	return new_list


# Example usage:
# file_path = 'rocstories-tr.txt'
# rows, token_rows = load_and_tokenize_text(file_path)

# Printing the first few rows and their tokenized words as an example
# for i in range(min(5, len(rows))):
#     print("Row:", i + 1)
#     print("Original:", rows[i])
#     print("Tokenized:", token_rows[i])
#     print()

# instances_tr = [sentence2instance(tokens, window) for tokens in token_rows]
# print(instances_tr[:3])

instances_tr, texts_tr = file2instance('rocstories-tr.txt', window, shuffle=True)
instances_te, texts_te = file2instance('rocstories-te.txt', window)

print(instances_tr[12])
print(texts_tr[12])

anchors_te = [list(instance['anchor'].keys())[0] for instance in instances_te]
instances_te_no_anchors = [{key: values for key, values in instance.items() if key != 'anchor'} for instance in instances_te]
# print(anchors_te)
# print(instances_te_no_anchors[:3])

for i in range(3):
	print(texts_tr[i])
	print(instances_tr[i])
	print("\n")

# Train:
tree = CobwebTree(0.000001, False, 0, True, False)
for i in range(len(instances_tr)):
	tree.ifit(instances_tr[i])
visualize(tree)

# Test:
test_profile = []
for i in range(len(instances_te)):
	profile = {
	'sequence': texts_te[i].replace(anchors_te[i], "___"),
	'true-anchor': anchors_te[i],
	}
	predict_probs = tree.categorize(instances_te_no_anchors[i]).predict_probs()
	pred_prob, anchor_pred = sorted([(prob, anchor) for (anchor, prob) in predict_probs['anchor'].items()], reverse=True)[0]
	profile['pred-anchor'] = anchor_pred
	profile['pred-prob'] = pred_prob
	profile['correct'] = int(anchor_pred == anchors_te[i])
	test_profile.append(profile)

with open('rocstories-test.csv', 'w', newline='', encoding='utf-8') as file:
	writer = csv.DictWriter(file, fieldnames=test_profile[0].keys())
	writer.writeheader()
	for profile in test_profile:
		writer.writerow(profile)






