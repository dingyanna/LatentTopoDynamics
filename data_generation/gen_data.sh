 
declare dyns=(epi popu gene eco2 wc lv)   

 
for dyn in "${dyns[@]}"; do
    for i in `seq 1 140`; do 
        python3 run.py --to_generate main --k 10 --dynamics $dyn --fix_dyn 0  --topology er  --n 100 --seed $i 
        python3 run.py --to_generate main --k 10 --dynamics $dyn --fix_dyn 0  --topology sf  --n 100 --seed $i     
        python3 run.py --to_generate main --k 10 --dynamics $dyn --fix_dyn 0  --topology com  --n 100 --seed $i         
    done   
done 
 

########### Generate out-of-distribution dataset
 
for dyn in "${dyns[@]}"; do 
    for i in `seq 21 100`; do 
        python3 run.py --n 100 --to_generate x0_out --x0_out 1 --dynamics $dyn --fix_dyn 0  --topology er  --k 10 --seed $i  
        python3 run.py --n 100 --to_generate x0_out --x0_out 1 --dynamics $dyn --fix_dyn 0  --topology sf  --k 10 --seed $i  
        python3 run.py --n 100 --to_generate x0_out --x0_out 1 --dynamics $dyn --fix_dyn 0  --topology com  --k 10 --seed $i  
        python3 run.py --n 100 --to_generate s_out --s_out 1 --dynamics $dyn --fix_dyn 0  --topology er  --k 10 --seed $i   
        python3 run.py --n 100 --to_generate s_out --s_out 1 --dynamics $dyn --fix_dyn 0  --topology sf  --k 10 --seed $i   
        python3 run.py --n 100 --to_generate s_out --s_out 1 --dynamics $dyn --fix_dyn 0  --topology com  --k 10 --seed $i   
        python3 run.py --n 100 --to_generate k_out --k_out 1 --dynamics $dyn --fix_dyn 0  --topology er --k 10 --seed $i    
        python3 run.py --n 100 --to_generate k_out --k_out 1 --dynamics $dyn --fix_dyn 0  --topology sf --k 10 --seed $i    
        python3 run.py --n 100 --to_generate k_out --k_out 1 --dynamics $dyn --fix_dyn 0  --topology com --k 10 --seed $i    
    done 
done

 

########### Generate a large network
 
declare dyns=(gene) 
for dyn in "${dyns[@]}"; do 
    for i in `seq 1 1`; do 
        #python3 run.py --n 100000 --to_generate large --dynamics $dyn --fix_dyn 0 --topology er  --k 12 --seed $i 
        python3 run.py --n 100000 --to_generate large_1 --dynamics $dyn --fix_dyn 0 --topology real --data PP-Pathways_ppi.edges.csv 
    done 
done
 
 