# python server.py --config-file configs/ncf16.yaml  --opt TRAIN.device cpu 
# python server.py --config-file configs/lora4_ncf16.yaml  --opt TRAIN.device cpu
# python server.py --config-file configs/lora8_ncf16.yaml  --opt TRAIN.device cpu DATALOADER.batch_size 128 TRAIN.lr 0.005
# python server.py --config-file configs/lora2_ncf16.yaml  --opt TRAIN.device cpu DATALOADER.batch_size 128 TRAIN.lr 0.005
python server.py --config-file configs/lora4_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003
python server.py --config-file configs/lora8_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003
python server.py --config-file configs/lora2_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003