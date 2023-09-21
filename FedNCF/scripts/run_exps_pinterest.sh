# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=5e-2 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=3e-1 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4

# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.aggregation=simpleavg
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=5e-2 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.aggregation=simpleavg
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=3e-1 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.aggregation=simpleavg

# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=1000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.local_epochs=4

# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=5e-2 FED.agg_epochs=1000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=5e-2 FED.agg_epochs=1000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.local_epochs=4

# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=3e-1 FED.agg_epochs=1000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=3e-1 FED.agg_epochs=1000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.local_epochs=4

python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=False TRAIN.lr=1e-1 FED.agg_epochs=550000 EVAL.interval=275000 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.num_clients=2 FED.aggregation=simpleavg