# python fedtrain.py data=lastfm net=fedmf net.init.gmf_emb_size=4 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=5e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16
# python fedtrain.py data=lastfm net=fedmf net.init.gmf_emb_size=8 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=5e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16

# python fedtrain.py data=lastfm net=fedmf net.init.gmf_emb_size=2 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=5e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16
# python fedtrain.py data=lastfm net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=5e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16 compression=svd FED.compression_kwargs.rank=2
# python fedtrain.py data=lastfm net=fedmf-lora-b net.init.gmf_emb_size=64 net.init.lora_rank=2 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=5e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16

# python fedtrain.py data=lastfm net=fedmf net.init.gmf_emb_size=16 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=5e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16
# python fedtrain.py data=lastfm net=fedmf-lora-b net.init.gmf_emb_size=64 net.init.lora_rank=16 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=5e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16
# python fedtrain.py data=lastfm net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep TRAIN.wandb=True TRAIN.lr=5e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=16 compression=svd FED.compression_kwargs.rank=16

python fedtrain.py data=lastfm net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep2 TRAIN.wandb=True TRAIN.lr=3e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=256
python fedtrain.py data=lastfm net=fedmf net.init.gmf_emb_size=64 task_name=fedsweep2 TRAIN.wandb=True TRAIN.lr=3e-3 FED.agg_epochs=1000 EVAL.interval=10 DATALOADER.batch_size=128