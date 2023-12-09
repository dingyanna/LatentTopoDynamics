declare niters=2
declare lr=0.01  
declare intval=100
declare bsize=10
declare pred=100
declare cond=10  
python3 run_models.py --dataset motion --random-seed 1 --ode-dims 16  --dynamics motion --gpu_id 1 --ode_type gcn --test_bsize $bsize --batch-size $bsize --lr $lr --optimizer AdamW --niters $niters --split_interval $intval --pred_length $pred --condition_length $cond
python3 run_models.py --dataset motion --random-seed 1 --ode-dims 16 --num_head 3 --dynamics motion --gpu_id 1 --ode_type gat --test_bsize $bsize --batch-size $bsize --lr $lr --optimizer AdamW --niters $niters --split_interval $intval --pred_length $pred --condition_length $cond
 