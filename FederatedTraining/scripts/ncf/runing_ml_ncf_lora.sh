python fedtrain.py data=ml-1m-v2 EXP.project=lora-fedrec-tuning  net=fedncf64-lora-fb net.init.lora_rank=16 task_name=fedsweep TRAIN.wandb=True FED.agg_epochs=2000 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=64 FED.local_epochs=2 TRAIN.weight_decay=1e-3 TRAIN.lr=1e-2
python fedtrain.py data=ml-1m-v2 EXP.project=lora-fedrec-tuning  net=fedncf64-lora-fb net.init.lora_rank=8 task_name=fedsweep TRAIN.wandb=True FED.agg_epochs=2000 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=64 FED.local_epochs=2 TRAIN.weight_decay=1e-3 TRAIN.lr=1e-2
python fedtrain.py data=ml-1m-v2 EXP.project=lora-fedrec-tuning  net=fedncf64-lora-fb net.init.lora_rank=4 task_name=fedsweep TRAIN.wandb=True FED.agg_epochs=2000 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=64 FED.local_epochs=2 TRAIN.weight_decay=5e-4 TRAIN.lr=1e-2
python fedtrain.py data=ml-1m-v2 EXP.project=lora-fedrec-tuning  net=fedncf64-lora-fb net.init.lora_rank=2 task_name=fedsweep TRAIN.wandb=True FED.agg_epochs=2000 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=64 FED.local_epochs=2 TRAIN.weight_decay=5e-4 TRAIN.lr=1e-2
python fedtrain.py data=ml-1m-v2 EXP.project=lora-fedrec-tuning  net=fedncf64-lora-fb net.init.lora_rank=32 task_name=fedsweep TRAIN.wandb=True FED.agg_epochs=2000 TRAIN.log_interval=10 EVAL.interval=100 DATALOADER.batch_size=64 FED.local_epochs=2 TRAIN.weight_decay=5e-4 TRAIN.lr=1e-2



