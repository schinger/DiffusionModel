# DiffusionModel
loss=0.0949
python dm.py --device cuda --learning_rate 2e-4 --dataset mnist --train_batch_size 128 --eval_batch_size 10 --num_epochs 100  --num_timesteps 1000 --embedding_size 100 --hidden_size 2048 --show_image_step 50

loss=0.0819
python dm.py --device cuda --learning_rate 2e-4 --dataset mnist --train_batch_size 128 --eval_batch_size 10 --num_epochs 100  --num_timesteps 1000 --embedding_size 100 --hidden_size 1000 --show_image_step 50


loss=0.108
python dm.py --device cuda --learning_rate 1e-4 --dataset mnist --train_batch_size 128 --eval_batch_size 10 --num_epochs 100  --num_timesteps 1000 --embedding_size 100 --hidden_size 2048 --show_image_step 50