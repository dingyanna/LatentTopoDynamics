# Data Generation

To generate sample dataset:
```
cd data_generation
./gen_data.sh
```
This sample dataset consists of 10 multivariate sequence. Each sequence contains 100 variables, corresponding to nodes in a graph. 

# Sample scripts

To train and test on gene regulatory dynamics
```
./sample.sh
```


To train and test on motion capture data
```
./motion.sh
```

The results are stored in `./results/date/model_config`, where `model_config` is specified in run_models.py.