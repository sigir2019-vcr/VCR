from __future__ import unicode_literals, print_function, division
import numpy as np
import random
import time
import datetime
import math

def load_data(TRAIN_DATA_PATH, TEST_DATA_PATH, TRACK_DATA_PATH):
	train = [line.strip() for line in open(TRAIN_DATA_PATH).readlines()]
	test = [line.strip() for line in open(TEST_DATA_PATH).readlines()]
	track_data = [line.strip() for line in open(TRACK_DATA_PATH).readlines()]

	return train, test, track_data

def make_dict(_train, _test, _track_data, maximum_title_length):
	user2id = {}
	id2user = []
	item2id = {"<PAD>": 0}
	id2item = ["<PAD>"]
	genre2id = {"<PAD>": 0}
	id2genre = ["<PAD>"]
	char2id = {"<PAD>": 0}
	id2char = ["<PAD>"]

	user2negative = {}
	track_data = {}

	# ============================
	# Dataset Format: 
	# Train: 
	#		user \t item_1 \t item_2 \t ....
	# Test:
	#		user \t correct \t negative_1 \t ...
	# ============================
	for line in _train:
		tokens = line.split("\t")
		user, items = tokens[0], tokens[1:]

		if user not in user2id:
			user2id[user] = len(user2id)
			id2user.append(user)
		
		for item in items:
			if item not in item2id:
				item2id[item] = len(item2id)
				id2item.append(item)

	for line in _test:
		tokens = line.split("\t")
		user, items = tokens[0], tokens[1:]

		if user not in user2id:
			user2id[user] = len(user2id)
			id2user.append(user)
		
		for item in items:
			if item not in item2id:
				item2id[item] = len(item2id)
				id2item.append(item)

	keys = set([i for i in range(len(item2id))])
	for line in _train:
		tokens = line.split("\t")
		user, items = tokens[0], tokens[1:]
		negatives = keys - set([item2id[item] for item in items])
		user2negative[user2id[user]] = negatives

	for line in _track_data:
		item, name, genre = line.split("\t")
		for char in name:
			if char not in char2id:
				char2id[char] = len(char2id)
				id2char.append(char)
		if genre not in genre2id:
			genre2id[genre] = len(genre2id)
			id2genre.append(genre)

		if item in item2id:
			if item2id[item] not in track_data:
				track_data[item2id[item]] \
				= [([char2id[char] for char in name[:maximum_title_length]]\
					+[0]*(maximum_title_length-len(name)), genre2id[genre])]
			else:
				track_data[item2id[item]]\
				.append(([char2id[char] for char in \
					name[:maximum_title_length]]\
					+[0]*(maximum_title_length-len(name)), genre2id[genre]))
	
	return user2id, id2user, item2id, id2item, genre2id, id2genre, \
	char2id, id2char, user2negative, track_data

def make_input(_train, _test, user2id, item2id):
	train, test = [], []

	for line in _train:
		tokens = line.split("\t")
		user, items = user2id[tokens[0]], [item2id[item] for item in tokens[1:]]
		for item in items:
			train.append((user, item))
	
	for line in _test:
		tokens = line.split("\t")
		user, candidates \
		= user2id[tokens[0]], [item2id[item] for item in tokens[1:]]
		test.append((user, candidates))

	return train, test

def add_padding(item2id, track_data, maximum_title_length):
	maximum_video_length = \
	max([len(tracks) for artist, tracks in track_data.items()])
	_track_data = {}

	for i in range(len(item2id)):
		if i not in track_data:
			_track_data[i] \
			= ([([0]*maximum_title_length, 0)]*maximum_video_length, 1)
		else:
			_track_data[i] \
			= ([(line[0]+[0]*(maximum_title_length - len(line[0])), line[1]) \
				for line in track_data[i]] + [([0]*maximum_title_length, 0, 1)]\
				*(maximum_video_length - len(track_data[i])), 
				len(track_data[i]))

	return _track_data, maximum_video_length

def train_batches(_train, batch_size, track_data, user2negative, sample_num):
	random.shuffle(_train)
	batch_num = int(len(_train)/batch_size) + 1

	for i in xrange(batch_num):
		users = []
		titles = []
		genres = []
		lenghts = []
		negative_titles = []
		negative_genres = []
		negative_lengths = []

		left = i*batch_size
		right = min((i+1)*batch_size, len(_train))

		for data in _train[left:right]:
			users.append(data[0])
			track_info = track_data[data[1]]
			titles.append([track[0] for track in track_info[0]])
			genres.append([track[1] for track in track_info[0]])
			lenghts.append(track_info[1])

			samples = [track_data[sample] for sample in \
			random.sample(user2negative[data[0]], sample_num)]
			negative_titles\
			.append([[track[0] for track in sample[0]] for sample in samples])
			negative_genres\
			.append([[track[1] for track in sample[0]] for sample in samples])
			negative_lengths.append([sample[1] for sample in samples])

		yield users, titles, genres, lenghts, negative_titles, \
		negative_genres, negative_lengths

def evaluation_batches(_test, batch_size, track_data):
	batch_num = int(len(_test)/batch_size) + 1

	for i in xrange(batch_num):
		users = []
		titles = []
		genres = []
		lenghts = []

		left = i*batch_size
		right = min((i+1)*batch_size, len(_test))

		for data in _test[left:right]:
			users.append(data[0])
			candidates = [track_data[candidate] for candidate in data[1]]
			titles.append([[track[0] for track in candidate[0]] \
				for candidate in candidates])
			genres.append([[track[1] for track in candidate[0]] \
				for candidate in candidates])
			lenghts.append([candidate[1] for candidate in candidates])

		yield users, titles, genres, lenghts

def get_ndcg(count):
	ndcg = [.0]*10
	for i in range(10):
		ndcg[i] += (1/math.log(i+2, 2))*count[i]

	return ndcg

def get_hit_ratio(prediction):
	hit_ratio = [.0]*10
	
	for i in range(len(prediction)):
		for j in range(10):
			if prediction[i][j] == 0:
				hit_ratio[j] += 1
				break
	return hit_ratio

def get_mrr(prediction):
	mrr = .0
	
	for i in range(len(prediction)):
		for j in range(len(prediction[i])):
			if prediction[i][j] == 0:
				mrr += (1.0/(j+1))
				break
				
	return mrr
