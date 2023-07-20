import argparse

# dataset name 
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

# model name 
model = 'NeuMF-end'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = './Data/'

train_rating = main_path + '{}.train.rating'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = './models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", 
        type=float, 
        default=0.001, 
        help="learning rate")
    parser.add_argument("--dropout", 
        type=float,
        default=0.0,  
        help="dropout rate")
    parser.add_argument("--batch_size", 
        type=int, 
        default=256, 
        help="batch size for training")
    parser.add_argument("--epochs", 
        type=int,
        default=20,  
        help="training epoches")
    parser.add_argument("--top_k", 
        type=int, 
        default=10, 
        help="compute metrics@top_k")
    parser.add_argument("--factor_num", 
        type=int,
        default=32, 
        help="predictive factors numbers in the model")
    parser.add_argument("--num_layers", 
        type=int,
        default=3, 
        help="number of layers in MLP model")
    parser.add_argument("--num_ng", 
        type=int,
        default=4, 
        help="sample negative items for training")
    parser.add_argument("--test_num_ng", 
        type=int,
        default=99, 
        help="sample part of negative items for testing")
    parser.add_argument("--out", 
        default=True,
        help="save model or not")
    parser.add_argument("--gpu", 
        type=str,
        default="0",  
        help="gpu card ID") 
    return parser
