import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

from operator import itemgetter
import math

"""
	PC-GNN Layers
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class InterAgg(nn.Module):
	# 跨关系聚合器

	def __init__(self, features, feature_dim, embed_dim, 
				 train_pos, adj_lists, intraggs, inter='GNN', cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes 所有node的feature或者embedding
		:param feature_dim: the input dimension
		:param embed_dim: the embed dimension  嵌入纬度
		:param train_pos: positive samples in training set 正样本
		:param adj_lists: a list of adjacency lists for each single-relation graph 每个single-relationship的邻接列表
		:param intraggs: the intra-relation aggregators used by each single-relation graph  单关系图的内部聚合器
		:param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'  聚合类型（为啥没用？）
		:param cuda: whether to use GPU
		"""
		super(InterAgg, self).__init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		# add intra_agg4 for alibaba
		#self.intra_agg4 = intraggs[3]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda
		# add intra_agg4 for alibaba
		#self.intra_agg4.cuda = cuda
		self.train_pos = train_pos

		# initial filtering thresholds
		# 设置了邻居节点的保留比例
		# 这里是对应了三种关系(yelpchi和amazon里面都是三种关系)
		self.thresholds = [0.5, 0.5, 0.5]

		# parameter used to transform node embeddings before inter-relation aggregation
		# 这一部分是layer内部的信息聚合
		# 跨关系聚合后，变为一个可训练的weight
		# self.feat_dim,每个节点的原始维度特征，输入到模型中的每个node的feature的大小
		# len(intraggs):内部聚合器的数量，每个聚合器对应一种关系类型的邻接信息处理， 这里表示有多少种不同的关系数据被用于节点嵌入
		self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim*len(intraggs)+self.feat_dim, self.embed_dim))
		# 这里weight的构成：
		# part1： 所有通过不同关系聚合得到的嵌入的总维度(每种关系产生的嵌入维度*关系数）
		# part2: 原始节点的维度
		init.xavier_uniform_(self.weight)

		# label predictor for similarity measure
		# 通过全连接层转换为二分类
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.relation_score_log = []

	def forward(self, nodes, labels, train_flag=True):
		"""
		:param nodes: a list of batch node ids
		:param labels: a list of batch node labels
		:param train_flag: indicates whether in training or testing mode
		:return combined: the embeddings of a batch of input node features
		:return center_scores: the label-aware scores of batch nodes
		"""

		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = []
		# 把每个node与对应的每种类型的邻接矩阵对应(一个node对应不止一个）
		for adj_list in self.adj_lists:
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		# find unique nodes and their neighbors used in current batch
		unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
								 set.union(*to_neighs[2]), set(nodes))

		# calculate label-aware scores
		if self.cuda:
			batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
			pos_features = self.features(torch.cuda.LongTensor(list(self.train_pos)))
		else:
			batch_features = self.features(torch.LongTensor(list(unique_nodes)))
			pos_features = self.features(torch.LongTensor(list(self.train_pos)))
		# 这里用linear全连接层处理批次节点的特征，输出每个节点属于各个类别的得分
		batch_scores = self.label_clf(batch_features)
		pos_scores = self.label_clf(pos_features)
		id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

		# the label-aware scores for current batch of nodes
		# 创建node*类别的得分matrix
		center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

		# get neighbor node id list for each batch node and relation
		# 三种关系列表
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]
		#r4_list = [list(to_neigh) for to_neigh in to_neighs[3]]

		# assign label-aware scores to neighbor nodes for each batch node and relation
		# 提取两个类别的对应得分
		r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list]
		r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list]
		r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]
		## add r4_scores for alibaba
		#r4_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r4_list]

		# count the number of neighbors kept for aggregation for each batch node and relation
		r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
		r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
		r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]
		## add r4_scores for alibaba
		#r4_sample_num_list = [math.ceil(len(neighs) * self.thresholds[3]) for neighs in r4_list]

		# intra-aggregation steps for each relation
		# Eq. (8) in the paper
		r1_feats, r1_scores = self.intra_agg1.forward(nodes, labels, r1_list, center_scores, r1_scores, pos_scores, r1_sample_num_list, train_flag)
		r2_feats, r2_scores = self.intra_agg2.forward(nodes, labels, r2_list, center_scores, r2_scores, pos_scores, r2_sample_num_list, train_flag)
		r3_feats, r3_scores = self.intra_agg3.forward(nodes, labels, r3_list, center_scores, r3_scores, pos_scores, r3_sample_num_list, train_flag)
		#r4_feats, r4_scores = self.intra_agg4.forward(nodes, labels, r4_list, center_scores, r4_scores, pos_scores,r4_sample_num_list, train_flag)

		# get features or embeddings for batch nodes
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		self_feats = self.features(index)

		# number of nodes in a batch
		n = len(nodes)

		# concat the intra-aggregated embeddings from each relation
		# Eq. (9) in the paper
		cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)
		#cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats, r4_feats), dim=1)

		combined = F.relu(cat_feats.mm(self.weight).t())

		# 1.返回当前层h_v^(l)的节点最终特征(包含节点自身特征，不同类型关系的邻居特征）
		# 2.返回节点的标签感知得分(属于各个类别的可能性）
		return combined, center_scores


class IntraAgg(nn.Module):

	def __init__(self, features, feat_dim, embed_dim, train_pos, rho, cuda=False):
		"""
		Initialize the intra-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feat_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param rho: the ratio of the oversample neighbors for the minority class
		# 少数类别的ratio，这里应该是要做多大的oversampling?
		:param cuda: whether to use GPU
		"""
		super(IntraAgg, self).__init__()

		self.features = features
		self.cuda = cuda
		self.feat_dim = feat_dim
		self.embed_dim = embed_dim
		self.train_pos = train_pos
		self.rho = rho
		# 权重矩阵: 整合node本身和邻居的特征。
		self.weight = nn.Parameter(torch.FloatTensor(2*self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes, batch_labels, to_neighs_list, batch_scores, neigh_scores, pos_scores, sample_list, train_flag):
		"""
		# 应用了graphsage的原理
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param nodes: list of nodes in a batch
		:param to_neighs_list: neighbor node id list for each batch node in one relation
		:param batch_scores: the label-aware scores of batch nodes 当前批次节点的label得分
		:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation #每个node的label的得分
		:param pos_scores: the label-aware scores 1-hop neighbors for the minority positive nodes #少数 pos node(的neighbor?)的label得分
		:param train_flag: indicates whether in training or testing mode # 是否为训练模式(是否更新参数）
		:param sample_list: the number of neighbors kept for each batch node in one relation
		:return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
		:return samp_scores: the average neighbor distances for each relation after filtering
		"""

		# filer neighbors under given relation in the train mode
		# 选择适当的模式挑选邻居node
		# 这里每个samp_neights 都是一个set
		if train_flag:
			samp_neighs, samp_scores = choose_step_neighs(batch_scores, batch_labels, neigh_scores, to_neighs_list, pos_scores, self.train_pos, sample_list, self.rho)
		else:
			# 这个是test用的neighbour
			samp_neighs, samp_scores = choose_step_test(batch_scores, neigh_scores, to_neighs_list, sample_list)
		
		# find the unique nodes among batch nodes and the filtered neighbors
		# 提取所有当前batch node的邻居nodes
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

		# intra-relation aggregation only with sampled neighbors
		# 这里mask聚合了所有可能的neighbor，但是只应用sampled neighbours
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)  # mean aggregator
		if self.cuda:
			self_feats = self.features(torch.LongTensor(nodes).cuda())
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			self_feats = self.features(torch.LongTensor(nodes))
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		agg_feats = mask.mm(embed_matrix)  # single relation aggregator
		cat_feats = torch.cat((self_feats, agg_feats), dim=1)  # concat with last layer
		to_feats = F.relu(cat_feats.mm(self.weight))
		return to_feats, samp_scores


def choose_step_neighs(center_scores, center_labels, neigh_scores, neighs_list, minor_scores, minor_list, sample_list, sample_rate):
    # 模型侧重于解决类不平衡问题
    """
    Choose step for neighborhood sampling
    :param center_scores: the label-aware scores of batch nodes
    :param center_labels: the label of batch nodes
    :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
    :param neighs_list: neighbor node id list for each batch node in one relation
	:param minor_scores: the label-aware scores for nodes of minority class in one relation
    :param minor_list: minority node id list for each batch node in one relation
    :param sample_list: the number of neighbors kept for each batch node in one relation
	:para sample_rate: the ratio of the oversample neighbors for the minority class
    """
    samp_neighs = []
    samp_score_diff = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score_neigh = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        # compute the L1-distance of batch nodes and their neighbors
        score_diff_neigh = torch.abs(center_score_neigh - neigh_score).squeeze()
        sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(score_diff_neigh, dim=0, descending=False)
        selected_neigh_indices = sorted_neigh_indices.tolist()

        # top-p sampling according to distance ranking
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_neigh_indices[:num_sample]]
            selected_score_diff = sorted_score_diff_neigh.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            selected_score_diff = score_diff_neigh.tolist()
            if isinstance(selected_score_diff, float):
                selected_score_diff = [selected_score_diff]

        if center_labels[idx] == 1:
            num_oversample = int(num_sample * sample_rate)
            center_score_minor = center_score.repeat(minor_scores.size()[0], 1)
            score_diff_minor = torch.abs(center_score_minor - minor_scores[:, 0].view(-1, 1)).squeeze()
            sorted_score_diff_minor, sorted_minor_indices = torch.sort(score_diff_minor, dim=0, descending=False)
            selected_minor_indices = sorted_minor_indices.tolist()
            selected_neighs.extend([minor_list[n] for n in selected_minor_indices[:num_oversample]])
            selected_score_diff.extend(sorted_score_diff_minor.tolist()[:num_oversample])

        samp_neighs.append(set(selected_neighs))
        samp_score_diff.append(selected_score_diff)

    return samp_neighs, samp_score_diff


def choose_step_test(center_scores, neigh_scores, neighs_list, sample_list):
	# 这个模型是test用的，采样时候不考虑label，不做过采样，侧重于模型在中立环境下的评估能力
	"""
	Filter neighbors according label predictor result with adaptive thresholds
	:param center_scores: the label-aware scores of batch nodes
	:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node in one relation
	:return samp_neighs: the neighbor indices and neighbor simi scores
	:return samp_scores: the average neighbor distances for each relation after filtering
	"""

	samp_neighs = []
	samp_scores = []
	for idx, center_score in enumerate(center_scores):
		center_score = center_scores[idx][0]
		neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
		center_score = center_score.repeat(neigh_score.size()[0], 1)
		neighs_indices = neighs_list[idx]
		num_sample = sample_list[idx]

		# compute the L1-distance of batch nodes and their neighbors
		score_diff = torch.abs(center_score - neigh_score).squeeze()
		sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
		selected_indices = sorted_indices.tolist()

		# top-p sampling according to distance ranking and thresholds
		if len(neigh_scores[idx]) > num_sample + 1:
			selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
			selected_scores = sorted_scores.tolist()[:num_sample]
		else:
			selected_neighs = neighs_indices
			selected_scores = score_diff.tolist()
			if isinstance(selected_scores, float):
				selected_scores = [selected_scores]

		samp_neighs.append(set(selected_neighs))
		samp_scores.append(selected_scores)

	return samp_neighs, samp_scores
