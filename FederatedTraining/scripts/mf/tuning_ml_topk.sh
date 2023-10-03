# Tuning lr and weight decay
python fedtrain.py -m EXP.project=lora-fedrec-tuning task_name=ml-topk4 TRAIN.wandb=True data=ml-1m-v2 net=fedmf net.init.gmf_emb_size=64 FED.agg_epochs=1500 DATALOADER.batch_size=64 FED.local_epochs=2 hparams_search=ml-optuna-lr-wd compression=topk FED.compression_kwargs.ratio=0.0636