# Tuning lr and weight decay
# python fedtrain.py -m EXP.project=lora-fedrec-tuning task_name=pinterest-base TRAIN.wandb=True data=pinterest-v2 net=fedmf net.init.gmf_emb_size=64 FED.agg_epochs=2000 hparams_search=pinterest-optuna

# 1/10 - ks
# python fedtrain.py -m EXP.project=lora-fedrec-tuning task_name=pinterest-base TRAIN.wandb=True data=pinterest-v2 net=fedmf net.init.gmf_emb_size=2 FED.agg_epochs=2000 hparams_search=pinterest-lr-wd DATALOADER.batch_size=128 FED.local_epochs=2
python fedtrain.py -m EXP.project=lora-fedrec-tuning task_name=pinterest-base TRAIN.wandb=True data=pinterest-v2 net=fedmf net.init.gmf_emb_size=8 FED.agg_epochs=2000 hparams_search=pinterest-lr-wd DATALOADER.batch_size=128 FED.local_epochs=2
python fedtrain.py -m EXP.project=lora-fedrec-tuning task_name=pinterest-base TRAIN.wandb=True data=pinterest-v2 net=fedmf net.init.gmf_emb_size=32 FED.agg_epochs=2000 hparams_search=pinterest-lr-wd DATALOADER.batch_size=128 FED.local_epochs=2
python fedtrain.py -m EXP.project=lora-fedrec-tuning task_name=pinterest-base TRAIN.wandb=True data=pinterest-v2 net=fedmf net.init.gmf_emb_size=16 FED.agg_epochs=2000 hparams_search=pinterest-lr-wd DATALOADER.batch_size=128 FED.local_epochs=2

