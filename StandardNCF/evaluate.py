import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


# def metrics(model, test_loader, top_k, device):
# 	HR, NDCG = [], []

# 	for user, item, label in test_loader:
# 		user = user.to(device)
# 		item = item.to(device)

# 		predictions = model(user, item)
# 		_, indices = torch.topk(predictions, top_k)
# 		recommends = torch.take(
# 				item, indices).cpu().numpy().tolist()

# 		gt_item = item[0].item()
# 		HR.append(hit(gt_item, recommends))
# 		NDCG.append(ndcg(gt_item, recommends))

# 	return np.mean(HR), np.mean(NDCG)


def metrics(model, test_loader, top_k, device='cpu', num_negatives=99):
	HR, NDCG = [], []

	preds = []
	for user, item, label in test_loader:
		user = user.to(device)
		item = item.to(device)
		predictions = model(user, item)
		preds.append(predictions)
	
	preds = torch.cat(preds, dim=0)
	preds = preds.view(-1, num_negatives + 1)
	_, topk_indices = torch.topk(preds, top_k, dim=-1)
	_tmp = topk_indices == 0
	HR = torch.any(_tmp, dim=-1).float().mean().item()
	_tmp = torch.argwhere(_tmp)[:, 1]
	NDCG = (torch.sum(torch.reciprocal(torch.log2(_tmp.float() + 2))) / preds.shape[0]).item()
	# _, indices = torch.topk(predictions, top_k)
	# recommends = torch.take(
	# 		item, indices).cpu().numpy().tolist()

	# gt_item = item[0].item()
	# HR.append(hit(gt_item, recommends))
	# NDCG.append(ndcg(gt_item, recommends))

	return HR, NDCG

