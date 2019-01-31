from __future__ import unicode_literals, print_function, division
import numpy as np
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import models
import utils
import sys
import time
import parameters

# =============================================
print("=====================================================================")
print("HYPER-PARAMETERS\n")
print("EMBEDDING_DIM: {}".format(parameters.embedding_dim))
print("SAMPLE_NUM: {}".format(parameters.sample_num))
print("EPOCHS: {}".format(parameters.epochs))
print("BATCH_SIZE: {}".format(parameters.batch_size))
print("LEARNING_RATE: {}".format(parameters.learning_rate))
print("DROPOUT_RATE: {}".format(parameters.dropout_rate))
print("L2_LAMBDA: {}".format(parameters.l2_lambda))
print("CANDIDATE_NUM: {}".format(parameters.candidate_num))

# =============================================
## DATA PREPARATION
print("=====================================================================")
print("Data Loading..")
train, test, track_data = utils.load_data(parameters.train_data_path, 
										  parameters.test_data_path, 
										  parameters.track_data_path)

print("Make Dictionary..")
user2id, id2user, item2id, id2item, genre2id, id2genre, char2id, id2char, \
user2negative, track_data = utils.make_dict(train, test, track_data, 
											parameters.maximum_title_length)

print("Make Input..")
train, test = utils.make_input(train, test, user2id, item2id)

print("Add Padding..")
track_data, maximum_video_length = utils.add_padding(
	item2id, 
	track_data, 
	parameters.maximum_title_length)

print("=====================================================================")
print("DATA STATISTICS\n")
print("TRAIN DATA: {}".format(len(train)))
print("TEST DATA: {}".format(len(test)))
print("NUMBER OF USERS: {}".format(len(user2id)))
print("NUMBER OF ITEMS: {}".format(len(item2id)))
print("NUMBER OF CHARS: {}".format(len(char2id)))
print("NUMBER OF GENRES: {}".format(len(genre2id)))
print("MAXIMUM_VIDEO_LENGTH: {}".format(maximum_video_length))
print("MAXIMUM_TITLE_LENGTH: {}".format(parameters.maximum_title_length))

print("=====================================================================")

model = models.VCR(parameters.embedding_dim, 
				   parameters.char_embedding_dim, 
				   parameters.genre_embedding_dim, 
				   len(user2id), 
				   len(item2id), 
				   len(char2id), 
				   len(genre2id), 
				   parameters.filter_num, 
				   parameters.filter_sizes, 
				   parameters.learning_rate, 
				   parameters.candidate_num, 
				   maximum_video_length, 
				   parameters.maximum_title_length).cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
					   lr=parameters.learning_rate)

best = [.0]*20
for i in xrange(parameters.epochs):
	model.train()
	step = 0
	loss = .0
	hit_ratio = [.0]*10
	batch_num = int(len(train)/parameters.batch_size) + 1
	start = time.time()
	batches = utils.train_batches(train, parameters.batch_size, 
								  track_data, user2negative, 
								  parameters.sample_num)
	for batch in batches:
		users, titles, genres, lengths, negative_titles, negative_genres, \
		negative_lengths = batch
		input_users = Variable(torch.cuda.LongTensor(users))
		input_titles = Variable(torch.cuda.LongTensor(titles))
		input_genres = Variable(torch.cuda.LongTensor(genres))
		input_lengths = Variable(torch.cuda.LongTensor(lengths))

		input_negative_titles \
		= Variable(torch.cuda.LongTensor(negative_titles))
		input_negative_genres \
		= Variable(torch.cuda.LongTensor(negative_genres))
		input_negative_lengths \
		= Variable(torch.cuda.LongTensor(negative_lengths))

		input_labels = Variable(torch.cuda.FloatTensor(
			([[1]+[0]*parameters.sample_num])*len(users)))

		# Optimizing
		optimizer.zero_grad()
		_loss = model(input_users, 
					  input_titles, 
					  input_genres, 
					  input_lengths, 
					  input_negative_titles, 
					  input_negative_genres, 
					  input_negative_lengths, 
					  input_labels)
		_loss.backward()
		optimizer.step()
		loss+=_loss.cpu().data.numpy()

		step += 1
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Process Training Epoch: [{}/{}] Batch: [{}/{}]"\
			.format(i+1, parameters.epochs, step, batch_num))

	if (i+1) % 1 == 0:
		model.eval()
		step = 0
		batch_num = int(len(test)/100) + 1

		batches = utils.evaluation_batches(test, 100, track_data)
		for batch in batches:
			users, titles, genres, lengths = batch
			input_users = Variable(torch.cuda.LongTensor(users))
			input_titles = Variable(torch.cuda.LongTensor(titles))
			input_genres = Variable(torch.cuda.LongTensor(genres))
			input_lengths = Variable(torch.cuda.LongTensor(lengths))

			scores = model.get_sample_scores(input_users, 
											 input_titles, 
											 input_genres, 
											 input_lengths)
			rank = model.evaluation(scores).cpu().data.numpy()

			batch_hit_ratio = utils.get_hit_ratio(rank)
			for j in range(10):
				hit_ratio[j] += batch_hit_ratio[j]			
			step += 1

			sys.stdout.write("\033[F")
			sys.stdout.write("\033[K")
			print("Process Evaluation Epoch: [{}/{}] Batch: [{}/{}]"\
				.format(i+1, parameters.epochs, step, batch_num))


		ndcg = utils.get_ndcg(hit_ratio)
		ndcg = [sum(ndcg[:j+1])/len(test) for j in range(len(ndcg))]
		hit_ratio \
		= [sum(hit_ratio[:j+1])/len(test) for j in range(len(hit_ratio))]

		if ndcg[9] >= best[9]:
			for j in range(0,10):
				best[j] = ndcg[j]
			for j in range(10,20):
				best[j] = hit_ratio[j-10]


		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("-------------------\nEpoch: [{}/{}] Loss: [{}] Time: [{}]\nNDCG: [{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}]\nHR: [{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}]\nBest: [{:.4f}/{:.4f}]\n".format(i+1, parameters.epochs, loss, time.time() - start, ndcg[0], ndcg[1], ndcg[2], ndcg[3], ndcg[4], ndcg[5], ndcg[6], ndcg[7], ndcg[8], ndcg[9], hit_ratio[0], hit_ratio[1], hit_ratio[2], hit_ratio[3], hit_ratio[4], hit_ratio[5], hit_ratio[6], hit_ratio[7], hit_ratio[8], hit_ratio[9], best[9], best[19]))

	else:
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Epoch: [{}/{}] Loss: {} Time: {}\n"\
			.format(i+1, parameters.epochs, loss, time.time() - start))
