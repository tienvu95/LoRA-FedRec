import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import data_utils
from tqdm import tqdm


args = config.get_parser().parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
device = torch.device("cuda")


############################## PREPARE DATASET ##########################
train_dataset = data_utils.MovieLen1MDataset(config.main_path, train=True, num_negatives=args.num_ng)
test_dataset = data_utils.MovieLen1MDataset(config.main_path, train=False)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

print("Num users", train_dataset.num_user)
print("Num items", train_dataset.num_item)
model = model.NCF(train_dataset.num_user, train_dataset.num_item, args.factor_num, args.num_layers, 
						args.dropout, config.model, GMF_model, MLP_model)
model.to(device)	
loss_function = nn.BCEWithLogitsLoss()

if config.model == 'NeuMF-pre':
	# optimizer = optim.SGD(model.parameters(), lr=args.lr)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

# writer = SummaryWriter() # for visualization

def log_time(fn):
	start = time.time()
	result = fn()
	return result, time.time() - start

########################### TRAINING #####################################
count, best_hr = 0, 0
for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).
	start_time = time.time()
	_, sample_time = log_time(lambda : train_loader.dataset.sample_negatives())
	# _, sample_time = log_time(lambda : train_loader.dataset.ng_sample())
	print("Sampling neg time", sample_time)
	for batch_idx, (user, item, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
		user = user.to(device)
		item = item.to(device)
		label = label.float().to(device)

		optimizer.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label)
		loss.backward()
		optimizer.step()
		# writer.add_scalar('data/loss', loss.item(), count)
		count += 1

	model.eval()
	HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)

	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		if args.out:
			if not os.path.exists(config.model_path):
				os.mkdir(config.model_path)
			torch.save(model, 
				'{}{}.pth'.format(config.model_path, config.model))

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
									best_epoch, best_hr, best_ndcg))
