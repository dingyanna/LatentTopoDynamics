 
declare dyns=(gene)   

 
for dyn in "${dyns[@]}"; do
    for i in `seq 1 10`; do 
        python3 run.py --to_generate sample --k 10 --dynamics $dyn --fix_dyn 0  --topology er  --n 100 --seed $i 
        python3 run.py --to_generate sample --k 10 --dynamics $dyn --fix_dyn 0  --topology sf  --n 100 --seed $i     
        python3 run.py --to_generate sample --k 10 --dynamics $dyn --fix_dyn 0  --topology com  --n 100 --seed $i         
    done   
done 
 