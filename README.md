# DiffusionModel
for mnist:
*******loss=0.036
python dm.py --device cuda --learning_rate 1e-3 --dataset mnist --train_batch_size 128 --eval_batch_size 10 --num_epochs 200  --num_timesteps 1000 --embedding_size 100 --hidden_size 2048 --hidden_layers 5 --show_image_step 50

load
python dm.py --device cuda --dataset mnist --eval_batch_size 10 --num_timesteps 1000 --embedding_size 100 --hidden_size 2048 --hidden_layers 5 --show_image_step 50 --eval_path exps/base/model.pth

----
2d data:
python dm.py

python dm.py --eval_path exps/base/model.pth