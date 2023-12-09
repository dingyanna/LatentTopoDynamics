declare niters=80
declare lr=0.01  
declare intval=100
declare bsize=10
declare pred=100
declare cond=25
declare dyns=(epi eco eco2 gene wc neural lv)  
declare topo_lis=(er sf com)
python3 run_models.py --dataset data_sample --random-seed 1 --ode-dims 16  --dynamics gene --gpu_id 1 --ode_type gcn --test_bsize $bsize --batch-size $bsize --lr $lr --optimizer AdamW --niters $niters --split_interval $intval --pred_length $pred --condition_length $cond
python3 run_models.py --dataset data_sample --random-seed 1 --ode-dims 16 --num_head 3 --dynamics gene --gpu_id 1 --ode_type gat --test_bsize $bsize --batch-size $bsize --lr $lr --optimizer AdamW --niters $niters --split_interval $intval --pred_length $pred --condition_length $cond
 