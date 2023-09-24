import numpy as np
import torch
from tqdm import tqdm


# def hit(gt_item, pred_items):
# 	if gt_item in pred_items:
# 		return 1
# 	return 0


# def ndcg(gt_item, pred_items):
# 	if gt_item in pred_items:
# 		index = pred_items.index(gt_item)
# 		return np.reciprocal(np.log2(index+2))
# 	return 0

def cal_loss(model, train_loader, loss_function, device='cpu'):
	losses = []
	for user, item, label in tqdm(train_loader, leave=False):
		user = user.to(device)
		item = item.to(device)
		label = label.to(device)
		predictions = model(user, item, mask_zero_user_index=False)
		loss = loss_function(predictions, label)
		losses.append(loss.item())
	return np.mean(losses).item()

def metrics(model, test_loader, top_k, device='cpu', num_negatives=99):
	preds = []
	for user, item, label in tqdm(test_loader, leave=False):
		user = user.to(device)
		item = item.to(device)
		predictions = model(user, item, mask_zero_user_index=False)
		preds.append(predictions)
	
	# Predefined true index
	true_index = num_negatives
	n_item_per_user = num_negatives + 1

	preds = torch.cat(preds, dim=0)
	preds = preds.view(-1, n_item_per_user)
	num_users = preds.shape[0]

	_, topk_indices = torch.topk(preds, top_k, dim=-1)
	is_hit = (topk_indices == true_index)
	user_hit = torch.any(is_hit, dim=-1)
	assert user_hit.shape[0] == preds.shape[0], str(user_hit.shape) + " " + str(preds.shape)
	HR = user_hit.float().mean().item()

	hit_rank = torch.argwhere(is_hit[user_hit])[:, 1]
	# Only one hit per user
	assert hit_rank.shape[0] == user_hit.sum(), str(hit_rank.shape) + " " + str(user_hit.sum())
	hit_rank = hit_rank + 1
	
	# Calculate the actual discounted gain for each record
	rel = 1
	discfun = torch.log2
	dcg = rel / discfun(hit_rank.float() + 1)

	NDCG = (torch.sum(dcg) / num_users).item()

	return HR, NDCG
