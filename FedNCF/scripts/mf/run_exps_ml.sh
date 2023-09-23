# python fedtrain.py data=ml-1m net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32
# python fedtrain.py data=ml-1m net=fedmf-lora-b net.init.gmf_emb_size=64 net.init.lora_rank=4 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32
# python fedtrain.py data=ml-1m net=fedmf-lora-b net.init.gmf_emb_size=64 net.init.lora_rank=8 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32
# python fedtrain.py data=ml-1m net=fedmf-lora-b net.init.gmf_emb_size=64 net.init.lora_rank=16 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32
# python fedtrain.py data=ml-1m net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32 compression=svd
# python fedtrain.py data=ml-1m net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32 compression=svd FED.compression_kwargs.rank=8

python fedtrain.py data=ml-1m net=fedmf net.init.gmf_emb_size=4 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32
python fedtrain.py data=ml-1m net=fedmf net.init.gmf_emb_size=8 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32
python fedtrain.py data=ml-1m net=fedmf net.init.gmf_emb_size=16 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32
python fedtrain.py data=ml-1m net=fedmf-lora-b net.init.gmf_emb_size=64 net.init.lora_rank=16 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32
python fedtrain.py data=ml-1m net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32 compression=svd FED.compression_kwargs.rank=16
python fedtrain.py data=ml-1m net=fedmf net.init.gmf_emb_size=2 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32
python fedtrain.py data=ml-1m net=fedmf-lora-b net.init.gmf_emb_size=64 net.init.lora_rank=2 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32
python fedtrain.py data=ml-1m net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=1e-2 FED.agg_epochs=400 EVAL.interval=10 DATALOADER.batch_size=32 compression=svd FED.compression_kwargs.rank=2


