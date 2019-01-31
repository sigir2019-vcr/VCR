from __future__ import unicode_literals, print_function, division
import torch
from torch.autograd import Variable
import torch.nn as nn

class VCR(nn.Module):
	def __init__(self, embedding_dim, 
					   char_embedding_dim, 
					   genre_embedding_dim, 
					   user_num, 
					   item_num, 
					   char_num, 
					   genre_num, 
					   filter_num, 
					   filter_sizes, 
					   learning_rate, 
					   candidate_num, 
					   maximum_video_length, 
					   maximum_title_length):
		
		super(VCR, self).__init__()

		# ==============================
		# Hyper-Parameters
		# ==============================
		self.embedding_dim = embedding_dim
		self.char_embedding_dim = char_embedding_dim
		self.genre_embedding_dim = genre_embedding_dim
		self.user_num = user_num
		self.item_num = item_num
		self.char_num = char_num
		self.genre_num = genre_num
		self.filter_num = filter_num
		self.filter_sizes = filter_sizes
		self.learning_rate = learning_rate
		self.candidate_num = candidate_num + 1
		self.maximum_video_length = maximum_video_length
		self.maximum_title_length = maximum_title_length

		# ==============================
		# Embeddings
		# ==============================
		self.user_embedding = nn.Embedding(self.user_num, self.embedding_dim)

		# ==============================
		# Char-CNN
		# ==============================
		self.cnn = Char_CNN(self.char_embedding_dim, self.char_num, 
							self.filter_num, self.filter_sizes, 
							self.maximum_title_length)

		# ==============================
		# RNN
		# ==============================
		self.rnn = 	RNN(self.embedding_dim, 
						self.genre_embedding_dim, 
						self.genre_num, 
						self.filter_num, 
						self.filter_sizes, 
						self.maximum_video_length)

		# ==============================
		# Output Layer
		# ==============================
		self.creterion = nn.BCEWithLogitsLoss()
		self.output_layer = nn.Linear(self.embedding_dim, 1)

	def forward(self, users, titles, genres, lengths, negative_titles, 
				negative_genres, negative_lenghts, labels):
		# ==============================
		# Output
		# ==============================
		positive = self.get_score(users,titles,genres,lengths)
		negatives = self.get_sample_scores(users, negative_titles, 
										   negative_genres, negative_lenghts)

		# ==============================
		# Loss
		# ==============================
		loss = self.creterion(torch.cat([positive, negatives], 1), labels)

		return loss

	def get_score(self, users, titles, genres, lengths):
		# ==============================
		# Embedding Lookup:  (batch, embedding_dim)
		# ==============================
		embedded_users = self.user_embedding(users)

		# Title Encoding: (batch, maximum_video_length, len(filter_sizes)*filter_num)
		# ==============================
		embedded_titles = self.title_encoding(titles)

		# ==============================
		# RNN
		# ==============================
		channel_vector, perm_index = self.rnn(embedded_titles, genres, lengths)

		# ==============================
		# Output
		# ==============================
		score = self.output_layer(embedded_users[perm_index]*channel_vector)

		return score

	def get_sample_scores(self, users, titles, genres, lengths):
		# ==============================
		# Output
		# ==============================
		titles = titles.transpose(0,1)
		genres = genres.transpose(0,1)
		lengths = lengths.transpose(0,1)

		scores = []
		for i in range(titles.size(0)):
			scores.append(self.get_score(users, titles[i], genres[i], 
										 lengths[i]).squeeze())
		
		scores = torch.stack(scores).transpose(0, 1)

		return scores

	def evaluation(self, scores):
		_, rank = torch.sort(scores, descending=True)
		return rank

	def title_encoding(self, titles):
		titles = titles.transpose(0, 1)
		embedded_titles = []
		for i in xrange(self.maximum_video_length):
			embedded_titles.append(self.cnn(titles[i]))
		embedded_titles = torch.stack(embedded_titles).transpose(0, 1)

		return  embedded_titles

class Char_CNN(nn.Module):
	def __init__(self, char_embedding_dim, char_num, filter_num, 
				 filter_sizes, maximum_title_length):
		super(Char_CNN, self).__init__()

		# ==============================
		# Hyper-Parameters
		# ==============================
		self.char_embedding_dim = char_embedding_dim
		self.char_num = char_num
		self.filter_num = filter_num
		self.filter_sizes = filter_sizes
		self.maximum_title_length = maximum_title_length

		# ==============================
		# Embeddings
		# ==============================
		self.char_embedding = nn.Embedding(self.char_num, 
										   self.char_embedding_dim)

		# ==============================
		# 1D CNN
		# ==============================
		self.cnn = nn.ModuleList([nn.Sequential(
			nn.Conv1d(self.char_embedding_dim, self.filter_num, size),
			nn.ReLU(),
			nn.MaxPool1d(self.maximum_title_length - size + 1)
			) for size in self.filter_sizes])


	def forward(self, input_titles):
		# ==============================
		# input_titles: (batch, maximum_title_length)
		# ==============================

		# ==============================
		# character embedding lookup: (batch, maximum_title_length, char_embedding_dim)
		# ==============================
		embedded_chars = self.char_embedding(input_titles).transpose(1,2)

		# ==============================
		# convolutions
		# ==============================
		convs = [conv(embedded_chars).squeeze() for conv in self.cnn]

		return torch.cat(convs, dim=1)


class RNN(nn.Module):
	def __init__(self, embedding_dim, genre_embedding_dim, genre_num, 
				 filter_num, filter_sizes, maximum_video_length):
		super(RNN, self).__init__()

		# ==============================
		# Hyper-Parameters
		# ==============================
		self.embedding_dim = embedding_dim
		self.input_size = filter_num*len(filter_sizes)
		self.genre_embedding_dim = genre_embedding_dim
		self.genre_num = genre_num
		self.maximum_video_length = maximum_video_length

		# ==============================
		# Embeddings
		# ==============================
		self.genre_embedding = nn.Embedding(self.genre_num, 
											self.genre_embedding_dim)

		# ==============================
		# RNN
		# ==============================
		self.rnn = nn.LSTM(self.input_size + self.genre_embedding_dim, 
						   self.embedding_dim, batch_first=True)


	def forward(self, input_titles, input_genres, lengths):
		# ==============================
		# input_titles: (batch, maximum_video_length, input_size)
		# input_genres: (batch, maximum_video_length, maximum_genre_length)
		# ==============================

		# ==============================
		# genre embedding lookup: (batch, maximum_video_length, genre_embedding_dim)
		# ==============================

		# input_genres = input_genres.transpose(0, 1)

		# embedded_genres = []
		# for i in xrange(self.maximum_video_length):
		# 	embedded_genres.append(self.genre_embedding(input_genres[i]).sum(1))
		# embedded_genres = torch.stack(embedded_genres).transpose(0, 1)

		embedded_genres = self.genre_embedding(input_genres)

		# ==============================
		# RNN
		# ==============================
		lengths, perm_index = lengths.sort(0, descending=True)

		input_titles = input_titles[perm_index]
		embedded_genres = embedded_genres[perm_index]

		packed_titles = torch.nn.utils.rnn.pack_padded_sequence(
			torch.cat([input_titles, embedded_genres], dim=2), 
					  list(lengths.data), batch_first=True)
		_, output = self.rnn(packed_titles)
	
		return output[0][0], perm_index






