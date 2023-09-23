# Tuning learning rate
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=5e-2 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=3e-1 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=5e-1 FED.agg_epochs=5000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=5e-1 FED.agg_epochs=5000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=5e-4



# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.aggregation=simpleavg
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=5e-2 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.aggregation=simpleavg
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=3e-1 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.aggregation=simpleavg

# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=1000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.local_epochs=4

# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=5e-2 FED.agg_epochs=1000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=5e-2 FED.agg_epochs=1000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.local_epochs=4

# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=3e-1 FED.agg_epochs=1000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=True TRAIN.lr=3e-1 FED.agg_epochs=1000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.local_epochs=4

# python fedtrain.py data=pinterest net=fedmf task_name=fedtrain4 TRAIN.wandb=False TRAIN.lr=1e-1 FED.agg_epochs=55000 EVAL.interval=27500 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4 FED.num_clients=2

# Run lora version
# lr = 1e-1
python fedtrain.py data=pinterest net=fedmf-lora-b net.init.lora_rank=4 task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=5000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=5e-4
python fedtrain.py data=pinterest net=fedmf-lora-b net.init.lora_rank=8 task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=5000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=5e-4
python fedtrain.py data=pinterest net=fedmf-lora-b net.init.lora_rank=16 task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=1e-1 FED.agg_epochs=5000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=5e-4

# lr = 1e-0.5 | 0.3162
# python fedtrain.py data=pinterest net=fedmf-lora-b net.init.lora_rank=4 task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=0.3162 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf-lora-b net.init.lora_rank=8 task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=0.3162 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4
# python fedtrain.py data=pinterest net=fedmf-lora-b net.init.lora_rank=16 task_name=fedtrain5 TRAIN.wandb=True TRAIN.lr=0.3162 FED.agg_epochs=2000 EVAL.interval=100 DATALOADER.batch_size=256 TRAIN.weight_decay=1e-4

