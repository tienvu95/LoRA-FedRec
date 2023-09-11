# python server.py --config-file configs/ncf16.yaml  --opt TRAIN.device cpu 
# python server.py --config-file configs/lora4_ncf16.yaml  --opt TRAIN.device cpu
# python server.py --config-file configs/lora8_ncf16.yaml  --opt TRAIN.device cpu DATALOADER.batch_size 128 TRAIN.lr 0.005
# python server.py --config-file configs/lora2_ncf16.yaml  --opt TRAIN.device cpu DATALOADER.batch_size 128 TRAIN.lr 0.005
# python server.py --config-file configs/lora4_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003
# python server.py --config-file configs/lora8_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003
# python server.py --config-file configs/lora2_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003
# python server.py --config-file configs/pinterest/lora4_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.01 FED.num_clients 200
# CUDA_VISIBLE_DEVICES=1 python server.py --config-file configs/pinterest/lora2_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.01 FED.num_clients 200
# CUDA_VISIBLE_DEVICES=1 python server.py --config-file configs/pinterest/lora8_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.01 FED.num_clients 200

# CUDA_VISIBLE_DEVICES=1 python server.py --config-file configs/pinterest/lora2_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.01 FED.num_clients 200
# CUDA_VISIBLE_DEVICES=1 python server.py --config-file configs/pinterest/lora8_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.01 FED.num_clients 200

# CUDA_VISIBLE_DEVICES=2 python server.py --config-file configs/foursq-ny/ncf16.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003 FED.num_clients 50
# CUDA_VISIBLE_DEVICES=2 python server.py --config-file configs/foursq-ny/lora4_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003 FED.num_clients 50
# CUDA_VISIBLE_DEVICES=2 python server.py --config-file configs/foursq-ny/lora2_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003 FED.num_clients 50
# CUDA_VISIBLE_DEVICES=2 python server.py --config-file configs/foursq-ny/lora8_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003 FED.num_clients 50
# CUDA_VISIBLE_DEVICES=2 python fedtrain.py data=4sq-ny net=fedmf-lora-b net.init.gmf_emb_size=64 net.init.lora_rank=4 task_name=debug TRAIN.lr=3e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16


CUDA_VISIBLE_DEVICES=2 python fedtrain.py data=4sq-ny net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=7e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16
CUDA_VISIBLE_DEVICES=2 python fedtrain.py data=4sq-ny net=fedmf-lora-b net.init.gmf_emb_size=64 net.init.lora_rank=4 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=7e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16
CUDA_VISIBLE_DEVICES=2 python fedtrain.py data=4sq-ny net=fedmf-lora-b net.init.gmf_emb_size=64 net.init.lora_rank=8 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=7e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16
# CUDA_VISIBLE_DEVICES=2 python fedtrain.py data=4sq-ny net=fedmf-lora-b net.init.gmf_emb_size=64 net.init.lora_rank=16 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=7e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16
CUDA_VISIBLE_DEVICES=2 python fedtrain.py data=4sq-ny net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=7e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16 compression=svd
CUDA_VISIBLE_DEVICES=2 python fedtrain.py data=4sq-ny net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=7e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16 compression=svd FED.compression_kwargrs.rank=8