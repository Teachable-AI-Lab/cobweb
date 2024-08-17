import os
import re
from multiprocessing import Pool

import spacy

# Import the sentence processor from spacy
nlp = spacy.load("en_core_web_sm", disable = ['parser'])
nlp.add_pipe("sentencizer")
nlp.max_length = float('inf')


""" The following functions are used for preprocessing in loading the stories """

def process_text(text, test=False):
	""" Load and preprocess a single (line) of text """
	# Preprocess
	if test:
		punc = re.compile(r"[^_a-zA-Z,.!?:;\s]")
	else:
		punc = re.compile(r"[^a-zA-Z,.!?:;\s]")
	whitespace = re.compile(r"\s+")
	text = punc.sub("", text)
	text = whitespace.sub(" ", text)
	text = text.strip().lower()
	
	# Parse
	text = nlp(text)
	text = [token.lemma_.lower() for token in text if (not token.is_punct and not token.is_stop)]

	return text


def process_file(idx, name, fp, verbose=True):
	""" Load and preprocess a text file """
	if verbose:
		print(f"Processing file {idx} - {name}")
	if not re.search(r'^[A-Z0-9]*.TXT$', name):
		return None
	with open(fp, encoding='latin-1') as fin:
		skip = True
		text = ""
		for line in fin:
			if not skip and "project gutenberg" not in line.lower():
				text += line
			elif "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS" in line:
				skip = False

		output = process_text(text)
		return output


def load_texts(training_dir, limit=None):
	""" Load the preprocessed texts used for training """
	for path, subdirs, files in os.walk(training_dir):
		if limit is None:
			limit = len(files)
		texts = [(idx, name, os.path.join(path, name)) for idx, name in enumerate(files[:limit])]

		# Preprocess the text files in parallel
		with Pool() as pool:
			outputs = pool.starmap(process_file, texts)
			for output in outputs:
				if output is None:
					continue
				yield output




""" The following functions are used for the instances (n-grams) generation: anchor + context words """

def get_instance(text, anchor_idx, anchor_wd, window):
	""" Generate an instance {'anchor': {anchor_word: 1}, 'context': {context_1: ..., context_2: ..., ...}} """
	before_text = text[max(0, anchor_idx - window):anchor_idx]
	after_text = text[anchor_idx + 1:anchor_idx + 1 + window]
	ctx_text = before_text + after_text
	ctx = {}

	# In a language task, the context words are not considered as simple counts.
	# Considering the proximity to the anchor word, the further the context word to the anchor, the less weight it will have
	for i, w in enumerate(before_text):
		ctx[w] = 1 / abs(len(before_text) - i)
	for i, w in enumerate(after_text):
		ctx[w] = 1 / (i + 1)

	instance = {}
	instance['context'] = ctx
	if anchor_wd is None:
		return instance
	instance['anchor'] = {anchor_wd: 1}
	return instance

def _story2instances(story, window):
	for anchor_idx, anchor_wd in enumerate(story):
		yield anchor_idx, get_instance(story, anchor_idx, anchor_wd, window=window)

def story2instances(story, window):
	return list(_story2instances(story, window))






