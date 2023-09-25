# python fedtrain.py data=ml-1m-v2 net=fedmf task_name=fedtrain TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=2000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=256 FED.local_epochs=1
python fedtrain.py data=ml-1m-v2 net=fedmf net.init.gmf_emb_size=32 task_name=fedtrain TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=2000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=256 FED.local_epochs=1
python fedtrain.py data=ml-1m-v2 net=fedmf net.init.gmf_emb_size=16 task_name=fedtrain TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=2000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=256 FED.local_epochs=1
python fedtrain.py data=ml-1m-v2 net=fedmf net.init.gmf_emb_size=4 task_name=fedtrain TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=2000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=256 FED.local_epochs=1
python fedtrain.py data=ml-1m-v2 net=fedmf net.init.gmf_emb_size=2 task_name=fedtrain TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=2000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=256 FED.local_epochs=1



