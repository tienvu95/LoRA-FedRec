# python fedtrain.py data=ml-1m-v2 net=fedmf task_name=fedsweep-v2 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=1000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1
python fedtrain.py data=ml-1m-v2 net=fedmf task_name=fedsweep-v2 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=1000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=256 FED.local_epochs=1

python fedtrain.py data=ml-1m-v2 net=fedmf task_name=fedsweep-v2 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=1000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=2
python fedtrain.py data=ml-1m-v2 net=fedmf task_name=fedsweep-v2 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=1000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=64 FED.local_epochs=2
python fedtrain.py data=ml-1m-v2 net=fedmf task_name=fedsweep-v2 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=1000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=128 FED.local_epochs=2
python fedtrain.py data=ml-1m-v2 net=fedmf task_name=fedsweep-v2 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=1000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=256 FED.local_epochs=2

python fedtrain.py data=ml-1m-v2 net=fedmf task_name=fedsweep-v2 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=1000 TRAIN.weight_decay=1e-4 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=256 FED.local_epochs=4
