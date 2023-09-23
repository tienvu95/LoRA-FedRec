# Tuning learning rate
# python fedtrain.py data=pinterest net=fedncf task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedncf task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=5e-2 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedncf task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=3e-1 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedncf task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=5e-1 FED.agg_epochs=5000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4

python fedtrain.py data=pinterest net=fedncf16 task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=5e-1 FED.agg_epochs=5000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=5e-4 TRAIN.log_interval=5
python fedtrain.py data=pinterest net=fedncf16 task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=5000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=5e-4 TRAIN.log_interval=5
python fedtrain.py data=pinterest net=fedncf16 task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=3e-1 FED.agg_epochs=5000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=5e-4 TRAIN.log_interval=5
python fedtrain.py data=pinterest net=fedncf16 task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=5e-2 FED.agg_epochs=5000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=5e-4 TRAIN.log_interval=5

python fedtrain.py data=pinterest net=fedncf16 task_name=debug TRAIN.wandb=False TRAIN.lr=5e-1 FED.agg_epochs=5000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=5e-4 TRAIN.log_interval=5