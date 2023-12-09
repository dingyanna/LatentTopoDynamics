import os
import sys
from lib.load_data import ParseData
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from torch.distributions.normal import Normal
from lib.create_coupled_ode_model import create_CoupledODE_model
from lib.utils import test_data_nn
import time 
import datetime
import calendar
import pandas as pd
import json 
import matplotlib.pyplot as plt 
#import wandb

parser = argparse.ArgumentParser('Coupled ODE')

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--pred_length', type=int, default=25, help="Number of days to predict ")
parser.add_argument('--condition_length', type=int, default=25, help="Number days to condition on")
 
parser.add_argument('--split_interval', type=int, default=3,
                    help="number of days between two adjacent starting date of two series.")

parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--lr', type=float, default=4e-3, help="Starting learning rate.")

parser.add_argument('-b', '--batch-size', type=int, default=10)
parser.add_argument('--test_bsize', type=int, default=10)

parser.add_argument('-r', '--random-seed', type=int, default=0, help="Random_seed")
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
#parser.add_argument('--edge_lamda', type=float, default=0.8, help='edge weight')
 
parser.add_argument('--rec-dims', type=int, default= 64, help="Dimensionality of the recognition model .")
parser.add_argument('--ode-dims', type=int, default=16, help="Dimensionality of the ODE func for edge and node (must be the same)")
parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in recognition model ")
#parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers  ODE func ")

parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')

 
 
# Encoder
parser.add_argument('--encoder', type=str, default='mlp')
parser.add_argument('--z0_n_heads', type=int, default=1)

# source task 
parser.add_argument('--ode_type', type=str, default='gcn')


parser.add_argument('--use_pe', type=int, default=0) 
parser.add_argument('--dynamics', type=str, default='gene')
 
parser.add_argument('--test_case', type=str, default='na', help='test scenario')


parser.add_argument('--num_head', type=int, default=1, help='number of attention heads in ODE')

parser.add_argument('--gpu_id', type=int, default=6, help='indicate whether degree is out-of-dist')
parser.add_argument('--topology', type=str, default='er')

args = parser.parse_args()


############ CPU AND GPU related
cuda_id = args.gpu_id
if torch.cuda.is_available():
	print("Using GPU" + "-"*80)
	device = torch.device(f"cuda:{cuda_id}")
else:
	print("Using CPU" + "-" * 80)
	device = torch.device("cpu") 
args.output_dim = 1 
if args.dataset == 'social':
    args.output_dim = 2
elif args.dataset == 'motion':
    args.output_dim = 6
#####################################################################################################
model_config = f"encoder_{args.encoder}_dyn_{args.dynamics}_topo_{args.topology}_ode_{args.ode_type}"
model_config = f"dim_{args.ode_dims}_cond_{args.condition_length}_ode_{args.ode_type}_seed_{args.random_seed}"
args.model_config = model_config
 

dataloader = ParseData(args =args, device=device)
train_encoder, train_decoder, train_batch, num_atoms, = dataloader.load_data(data_type="train")
val_encoder, val_decoder, val_batch, num_atoms = dataloader.load_data(data_type="val")
test_encoder, test_decoder, test_batch, num_atoms, = dataloader.load_data(data_type="test")
  
position_enc = 0 
args.num_atoms = num_atoms 
args.num_edges = dataloader.num_edges 
args.num_train_batch = train_batch
args.num_val_batch = val_batch
mlp_idx = 0
 
def main():
      
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    total_gradient_norm = [] 

    #Saving Path
    file_name = os.path.basename(__file__)[:-3]  # run_models
    utils.makedirs(args.save)
    experimentID = int(SystemRandom().random() * 100000)

    #Command Log
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command) 

    #Loading Data
     
    input_dim = dataloader.num_features 

    print('\nnum atoms', num_atoms) 
    print('batch size', dataloader.batch_size)
    print('\n--------------------------------\n') 

    # Model Setup
    # Create the model
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    
    model = create_CoupledODE_model(args, input_dim, z0_prior, obsrv_std, device)
    print('\nCreate Model') 
    count = 0
    for n,m in model.named_modules():
        count += np.sum([p.numel() for p in m.parameters(recurse=False)]).item()
    print('# Params', count)

    print('\n--------------------------------\n')
      
    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        utils.get_ckpt_model(ckpt_path, model, device)
        print("loaded saved ckpt!")
        #exit() 

    # Training Setup 
    args.save_path = model_config
    today = datetime.datetime.now()
    date = f"{calendar.month_name[today.month][:3]}{today.day}"
    args.date = date 
    if not os.path.exists(f"./logs/{date}/{model_config}"):
        os.makedirs(f"./logs/{date}/{model_config}")
    if not os.path.exists(f"./results/{date}/{model_config}"):
        os.makedirs(f"./results/{date}/{model_config}")
    if not os.path.exists(f"./checkpoints/{date}/{model_config}"):
        os.makedirs(f"./checkpoints/{date}/{model_config}")

    with open(f"./results/{date}/{model_config}/config.json", "w") as json_file:
        json.dump(vars(args), json_file, indent=4)    
     
    log_path = f"./logs/{date}/{model_config}/" + "result.log"
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    # Optimizer
    if args.optimizer == "AdamW":
        optimizer =optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr 
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)


    wait_until_kl_inc = 10
    best_test_MAPE = np.inf
    best_test_RMSE = np.inf
    best_val_MAPE = np.inf
    best_val_RMSE = np.inf
    n_iters_to_viz = 25
     

    def train_single_batch(model,batch_dict_encoder,batch_dict_decoder ,kl_coef):

        optimizer.zero_grad()
        train_res, pred_node  = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, args, train_encoder, kl_coef=kl_coef,istest=False, pe=position_enc)
        loss = train_res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
         
        optimizer.step()

        #loss_value = loss.data.item()

        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        return train_res, pred_node 

    def train_epoch(epo):
        model.train()
        loss_list = [] 
        loss_state = []
        MAPE_list = [] 
        MAPE_all = [] # number of batches x batch size x num nodes x seq length
         
        total_time = 0
        total_encoder_time = 0
        total_ode_time = 0  
        torch.cuda.empty_cache()
        train_pred_traj = []
        true_traj = []
        pred_loss_lis = []
        true_loss_lis = []
        for itr in tqdm(range(train_batch)):

            #utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr / 10)
            wait_until_kl_inc = 1000

            if itr < wait_until_kl_inc:
                kl_coef = 1
            else:
                kl_coef = 1*(1 - 0.99 ** (itr - wait_until_kl_inc))

            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device)
            #batch_dict_source_data = utils.get_next_batch_new(train_source_data, device) 
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)
              
            train_res, pred_node  = train_single_batch(model,batch_dict_encoder,batch_dict_decoder,kl_coef)
             
            #saving results
            loss_list.append(train_res['loss'].data.item()) 
            loss_state.append(train_res["loss_state"].data.item())
            MAPE_list.append(train_res["MAPE"])    
            MAPE_all.append(train_res["MAPE_ALL"]) 
            #print('\n\ntrain_res["MAPE_ALL"]', train_res["MAPE_ALL"].shape, '\n\n')     
            total_encoder_time += train_res["encoder_time"]
            total_ode_time += train_res["ode_time"] 
            train_pred_traj.append(pred_node)
            true_traj.append(batch_dict_decoder["data"].detach().cpu().numpy())
             
            #del batch_dict_encoder, batch_dict_source_data, batch_dict_decoder
                #train_res, loss
            torch.cuda.empty_cache()

        scheduler.step()  
        # message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | MAPE {:.6F} | Edge MAPE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
        #     epo,
        #     np.mean(loss_list), np.mean(MAPE_list),np.mean(EdgeMAPE_list), np.mean(likelihood_list),
        #     np.mean(kl_first_p_list), np.mean(std_first_p_list))
        message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | MAPE {:.6F} | Loss Curve Error {:.6F} | ODE time {:.4F}'.format(
            epo,
            np.mean(loss_list), 
            np.mean(MAPE_list),
            np.mean(loss_state),    
            total_ode_time / train_batch )
        MAPE_all = np.concatenate(MAPE_all, axis=0)
        #print('\n Train MAPE_ALL shape ', MAPE_all.shape, '\n')
        res = {
            "loss": np.mean(loss_list),  
            "loss_state": np.mean(loss_state),
            "mape": np.mean(MAPE_list),   
            "encoder_time": total_encoder_time / train_batch,
            "ode_time": total_ode_time / train_batch,  
        } 
        return message_train,kl_coef, res, np.concatenate(train_pred_traj, axis=0), np.concatenate(true_traj, axis=0), MAPE_all 

    def val_epoch(epo,kl_coef):
        model.eval()
        MAPE_list = [] 
        loss_list = []
        MAPE_all = []
        val_pred_traj = []
        true_traj = []
        pred_loss_lis = []
        true_loss_lis = []

        ode_time = []
        torch.cuda.empty_cache() 
        for itr in tqdm(range(val_batch)):
            batch_dict_encoder = utils.get_next_batch_new(val_encoder, device)
            #batch_dict_source_data = utils.get_next_batch_new(val_source_data, device) 
            batch_dict_decoder = utils.get_next_batch(val_decoder, device)
                        
            val_res, pred_node  = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder,  args, val_encoder, kl_coef=kl_coef,istest=True, pe=position_enc)
            val_pred_traj.append(pred_node)
            true_traj.append(batch_dict_decoder["data"].detach().cpu().numpy())
             
            ode_time.append(val_res["ode_time"])
            MAPE_list.append(val_res["MAPE"]), loss_list.append(val_res["loss"].data.item())
            MAPE_all.append(val_res["MAPE_ALL"])
            #del batch_dict_encoder, batch_dict_source_data, batch_dict_decoder
            # train_res, loss
            torch.cuda.empty_cache()

        MAPE_all = np.concatenate(MAPE_all, axis=0)
        #print('\n Val MAPE_ALL shape ', MAPE_all.shape, '\n') # number of sequences x num nodes x seq len
        message_val = 'Epoch {:04d} [Val seq (cond on sampled tp)] |  Loss {:.6F} |  MAPE {:.6F} | ODE Time {:.6F}'.format(
            epo,
            np.mean(loss_list),
            np.mean(MAPE_list),
            np.mean(ode_time) ) 
        return message_val, np.mean(MAPE_list),np.mean(loss_list), np.concatenate(val_pred_traj, axis=0), np.concatenate(true_traj, axis=0), MAPE_all, np.mean(ode_time) 

    # Test once: for loaded model
    if args.load is not None:
        test_res, MAPE_each, RMSE_each = test_data_nn(model, args.pred_length, args.condition_length, dataloader,
                                                   device=device, args=args, kl_coef=0)

        message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | MAPE {:.6F} | RMSE {:.6F} | KL fp {:.4f} | FP STD {:.4f}|'.format(
            0,
            test_res["loss"], test_res["MAPE"], test_res["RMSE"] ,
            test_res["kl_first_p"], test_res["std_first_p"])

        logger.info("Experiment " + str(experimentID))
        logger.info(message_test)
        logger.info(MAPE_each)
        logger.info(RMSE_each)
    

    def init_eval( ):
        model.eval()
        TrainMAPE_list = [] 
        loss_list = [] 
        
        train_true_traj = []
        train_pred_traj = []
        
        for itr in tqdm(range(train_batch)): 
            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device) 
            batch_dict_decoder = utils.get_next_batch(train_decoder, device) 
            train_res, pred_node, pred_loss, true_loss = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder,  args, val_encoder, kl_coef=1, pe=position_enc)
            TrainMAPE_list.append(train_res["MAPE_ALL"])           
            train_pred_traj.append(pred_node)
            train_true_traj.append(batch_dict_decoder["data"].detach().cpu().numpy()) 
            torch.cuda.empty_cache()

        MAPE_list = [] 
        val_pred_traj = []
        true_traj = [] 
        for itr in tqdm(range(val_batch)):
            batch_dict_encoder = utils.get_next_batch_new(val_encoder, device) 
            batch_dict_decoder = utils.get_next_batch(val_decoder, device) 
            val_res, pred_node, pred_loss, true_loss = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder,  args, val_encoder, kl_coef=1,istest=True, pe=position_enc )
            val_pred_traj.append(pred_node)
            true_traj.append(batch_dict_decoder["data"].detach().cpu().numpy())

            MAPE_list.append(val_res['MAPE_ALL']), loss_list.append(val_res["loss"].data.item()) 
            #del batch_dict_encoder, batch_dict_source_data, batch_dict_decoder
            # train_res, loss
            torch.cuda.empty_cache()
        val_pred_traj = np.concatenate(val_pred_traj, axis=0)
        val_true_traj = np.concatenate(true_traj, axis=0) 
        train_pred_traj = np.concatenate(train_pred_traj,axis=0)
        train_true_traj = np.concatenate(train_true_traj, axis=0)

        np.save(f"./results/{date}/{model_config}/init_train_pred_traj_{args.random_seed}.npy", train_pred_traj)
        np.save(f"./results/{date}/{model_config}/init_train_true_traj_{args.random_seed}.npy", train_true_traj)
        np.save(f"./results/{date}/{model_config}/init_val_pred_traj_{args.random_seed}.npy", val_pred_traj)
        np.save(f"./results/{date}/{model_config}/init_val_true_traj_{args.random_seed}.npy", val_true_traj) 
        np.save(f"./results/{date}/{model_config}/init_train_mape_{args.random_seed}.npy", np.concatenate(TrainMAPE_list, axis=0))
        np.save(f"./results/{date}/{model_config}/init_val_mape_{args.random_seed}.npy", np.concatenate(MAPE_list, axis=0))

    
    # Training and Testing
    t1 = time.time()
    best_err = np.inf 
    best_val_err = np.inf 
    train_loss_list = [] 
    train_loss_state_list = []
    train_mape_list = []
    val_loss_list = []
    val_mape_list = []
    val_ode_time = []
    #init_eval()
    min_val_loss_idx = 0
  
    for epo in range(1, args.niters + 1):
        args.epoch = epo
        message_train, kl_coef, train_res, train_pred_traj, train_true_traj, train_mape_all  = train_epoch(epo)
        message_val, mape_val, loss_val, val_pred_traj, val_true_traj, val_mape_all, ode_time = val_epoch(epo,kl_coef) 
        
        train_loss_list.append(train_res["loss"])
        train_mape_list.append(train_res["mape"]) 
        train_loss_state_list.append(train_res["loss_state"])
        val_loss_list.append(loss_val)
        val_mape_list.append(mape_val)
        val_ode_time.append(ode_time)

        logger.info("Experiment " + str(experimentID))
        logger.info(message_train)
        logger.info(message_val)
        gradient_norm = 0.0
        for param in model.parameters():
            if param.grad is None: 
                continue 
            gradient_norm += param.grad.data.norm(2).item() ** 2  
        gradient_norm = gradient_norm ** 0.5 
        total_gradient_norm.append(gradient_norm)
        if gradient_norm < 1e-6:
            break 
        # if min_val_loss_idx + 20 < epo:
        #     logger.info("Validation loss has not improved for 20 steps!")
        #     logger.info("Early stopping applies.")
        #     break    
        # Store model and predicted node value.
 

        if train_res["loss"] < best_err:
            best_err = train_res["loss"] 
            rmse = np.sqrt(np.mean((train_pred_traj - train_true_traj) ** 2, 0).mean(0))
            fig, axs = plt.subplots(1, 2, figsize=(6,3))
            axs[0].plot(rmse)
            axs[1].plot(train_mape_all.mean(0).mean(0))
            axs[0].set_xlabel('T')
            axs[0].set_ylabel('RMSE')
            axs[1].set_xlabel('T')
            axs[1].set_ylabel('MAPE')
            fig.tight_layout()
            plt.savefig(f"./results/{date}/{model_config}/{args.dynamics}_extrap_train.jpg", dpi=1200, format="jpg", bbox_inches='tight')
            
            if args.dataset == 'motion':
                fig, axs = plt.subplots(3,4, figsize=(12,9)) 
            else:
                fig, axs = plt.subplots(1,4, figsize=(12,3))  
            for j in range(min(50,train_true_traj.shape[1])):  
                if train_true_traj.shape[-1] == 1: 
                    axs[0].plot(train_true_traj[mlp_idx, j], color='tab:blue', linewidth=1 )
                    axs[1].plot(train_pred_traj[mlp_idx, j], color='tab:orange', linewidth=1 )
                    axs[2].plot(train_true_traj[mlp_idx, -1-j], color='tab:blue', linewidth=1 )
                    axs[3].plot(train_pred_traj[mlp_idx, -1-j], color='tab:orange', linewidth=1 )
                elif args.dataset == 'social':
                    axs[0].plot(train_true_traj[mlp_idx, j, :, 0], color='tab:blue', linewidth=1 )
                    axs[1].plot(train_pred_traj[mlp_idx, j, :, 0], color='tab:orange', linewidth=1 )
                    axs[2].plot(train_true_traj[mlp_idx, j, :, 1], color='tab:blue', linewidth=1 )
                    axs[3].plot(train_pred_traj[mlp_idx, j, :, 1], color='tab:orange', linewidth=1 )
                else: 
                    axs[0,0].plot(train_true_traj[mlp_idx, j, :, 0], color='tab:blue', linewidth=1 )
                    axs[0,1].plot(train_pred_traj[mlp_idx, j, :, 0], color='tab:orange', linewidth=1 )
                    axs[0,2].plot(train_true_traj[mlp_idx, j, :, 1], color='tab:blue', linewidth=1 )
                    axs[0,3].plot(train_pred_traj[mlp_idx, j, :, 1], color='tab:orange', linewidth=1 )
                    
                    axs[1,0].plot(train_true_traj[mlp_idx, j, :, 2], color='tab:blue', linewidth=1 )
                    axs[1,1].plot(train_pred_traj[mlp_idx, j, :, 2], color='tab:orange', linewidth=1 )
                    axs[1,2].plot(train_true_traj[mlp_idx, j, :, 3], color='tab:blue', linewidth=1 )
                    axs[1,3].plot(train_pred_traj[mlp_idx, j, :, 3], color='tab:orange', linewidth=1 )
                    
                    axs[2,0].plot(train_true_traj[mlp_idx, j, :, 4], color='tab:blue', linewidth=1 )
                    axs[2,1].plot(train_pred_traj[mlp_idx, j, :, 4], color='tab:orange', linewidth=1 )
                    axs[2,2].plot(train_true_traj[mlp_idx, j, :, 5], color='tab:blue', linewidth=1 )
                    axs[2,3].plot(train_pred_traj[mlp_idx, j, :, 5], color='tab:orange', linewidth=1 )
 
            fig.tight_layout()
            plt.savefig(f"./results/{date}/{model_config}/{args.dynamics}_traj_train_mlp{mlp_idx}.jpg", dpi=1200, format="jpg", bbox_inches='tight')
            plt.close()
            np.save(f"./results/{date}/{model_config}/{args.dynamics}_train_pred_traj_{args.random_seed}.npy", train_pred_traj)
            np.save(f"./results/{date}/{model_config}/{args.dynamics}_train_true_traj_{args.random_seed}.npy", train_true_traj) 
            np.save(f"./results/{date}/{model_config}/{args.dynamics}_train_mape_{args.random_seed}.npy", train_mape_all) 
        
        if loss_val < best_val_err:
            best_val_err = loss_val
            min_val_loss_idx = epo 
            np.save(f"./results/{date}/{model_config}/{args.dynamics}_val_pred_traj_{args.random_seed}.npy", val_pred_traj)
            np.save(f"./results/{date}/{model_config}/{args.dynamics}_val_true_traj_{args.random_seed}.npy", val_true_traj) 
            np.save(f"./results/{date}/{model_config}/{args.dynamics}_val_mape_{args.random_seed}.npy", val_mape_all) 
            ckpt_path = f"./checkpoints/{date}/{model_config}/saved_model_{args.random_seed}.ckpt"
            torch.save({
                        'args': args,
                        'state_dict': model.state_dict(),
                    }, ckpt_path)  
        if epo > 0 and epo % n_iters_to_viz == 0:
            test()
            
        torch.cuda.empty_cache() 
        plt.close()
    t2 = time.time()
    logger.info(f'Runtime {t2 - t1}')  
    pd.DataFrame({
        'train_loss': train_loss_list, 
        'train_loss_state': train_loss_state_list,
        'train_mape': train_mape_list,  
        'val_loss': val_loss_list,
        'val_mape': val_mape_list,
        'val_ode_time': val_ode_time
    }).to_csv(f'./results/{date}/{model_config}/train_curve_{args.random_seed}.csv')

    pd.DataFrame({
        'grad_norm': total_gradient_norm 
    }).to_csv(f'./results/{date}/{model_config}/grad_norm.csv')
 
    res = {
        #'train_loss': np.min(train_loss_list),
        #'val_loss': np.min(train_loss_list),
        'train_mape': np.min(train_mape_list),  
        'val_mape': np.min(val_mape_list)
    }
    return res

def test(): 
    dyn_name = {
        "eco": "Mutualistic",
        "gene": "Regulatory",
        "epi": "SIS",
        "ko": "Kuramoto",
        "wc": "Wilson-Cowan",
        "lv": "Lotka-Volterra",
        "eco2": "Mutualistic2",
        "gene2": "Regulatory2",
        "neural": "Neural",
        "popu": "Population"
    }
    args.save_path = model_config
    today = datetime.datetime.now()
    date = f"{calendar.month_name[today.month][:3]}{today.day}"
     
    args.date = date 

    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
 
    model = create_CoupledODE_model(args, dataloader.num_features, z0_prior, obsrv_std, device)
    
    ckpt_path = f"./checkpoints/{date}/{model_config}/saved_model_{args.random_seed}.ckpt"
    utils.get_ckpt_model(ckpt_path,model , device)
    print("loaded saved ckpt!")

    model.eval()
    MAPE_list = [] 
    loss_list = [] 
    val_pred_traj = []
    true_traj = []
     
    ode_time = []
    torch.cuda.empty_cache() 
    for itr in tqdm(range(test_batch)):
        batch_dict_encoder = utils.get_next_batch_new(test_encoder, device) 
        batch_dict_decoder = utils.get_next_batch(test_decoder, device)
                    
        val_res, pred_node = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder,  args, test_encoder, kl_coef=0, istest=True, pe=position_enc)
        val_pred_traj.append(pred_node)
        true_traj.append(batch_dict_decoder["data"].detach().cpu().numpy())
       
        ode_time.append(val_res["ode_time"])
        MAPE_list.append(val_res["MAPE"] ), loss_list.append(val_res["loss"].data.item())
         
        torch.cuda.empty_cache()
     
    #print('\n Val MAPE_ALL shape ', MAPE_all.shape, '\n') # number of sequences x num nodes x seq len
    message_val = ' [test seq (cond on sampled tp)] |  Loss {:.6F} |  MAPE {:.6F} | ODE Time {:.6F}'.format( 
        np.mean(loss_list),
        np.mean(MAPE_list),
        np.mean(ode_time) ) 
    print(message_val)
    test_pred_traj = np.concatenate(val_pred_traj, axis=0)
    test_true_traj = np.concatenate(true_traj, axis=0) 
    MAPE_all =  (np.abs((test_pred_traj - test_true_traj ) / test_true_traj )) 
    rmse_all = np.sqrt(np.mean((test_pred_traj - test_true_traj) ** 2))
    print('RMSE', rmse_all)
    
    rmse = np.sqrt(np.mean((test_pred_traj - test_true_traj) ** 2, 0).mean(0))
    print('rmse.shape', rmse.shape)
    
    fig, axs = plt.subplots(1, 2, figsize=(6,3))
    axs[0].plot(rmse)
    axs[1].plot(MAPE_all.mean(0).mean(0))
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('RMSE')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('MAPE')
    fig.tight_layout()
    plt.savefig(f"./results/{date}/{model_config}/{args.dynamics}_extrap_test.jpg", dpi=1200, format="jpg", bbox_inches='tight')
    
    
    np.save(f"./results/{date}/{model_config}/{args.dynamics}_test_pred_traj_{args.random_seed}.npy", test_pred_traj)
    np.save(f"./results/{date}/{model_config}/{args.dynamics}_test_true_traj_{args.random_seed}.npy", test_true_traj) 
    np.save(f"./results/{date}/{model_config}/{args.dynamics}_test_mape_{args.random_seed}.npy", MAPE_all)
    
    plt.clf()
 
    if args.dataset == 'social':
        fig, axs = plt.subplots(2,4,figsize=(6,4))
        T = [0, 10, test_true_traj.shape[2]//2, -1 ]
        for t in range(len(T)):
            axs[0,t].plot(test_true_traj[:,:,T[t], 0].reshape(-1), test_pred_traj[:,:,T[t], 0].reshape(-1), '.', color='tab:blue')
            axs[1,t].plot(test_true_traj[:,:,T[t], 1].reshape(-1), test_pred_traj[:,:,T[t], 1].reshape(-1), '.', color='tab:blue')
             
    elif args.dataset == 'motion':
        fig, axs = plt.subplots(6,4,figsize=(8,12))
        T = [0, 5, test_true_traj.shape[2]//2, -1 ]
        for t in range(len(T)):
            axs[0,t].plot(test_true_traj[:,:,T[t], 0].reshape(-1), test_pred_traj[:,:,T[t], 0].reshape(-1), '.', color='tab:blue')
            axs[1,t].plot(test_true_traj[:,:,T[t], 1].reshape(-1), test_pred_traj[:,:,T[t], 1].reshape(-1), '.', color='tab:blue')
            axs[2,t].plot(test_true_traj[:,:,T[t], 2].reshape(-1), test_pred_traj[:,:,T[t], 2].reshape(-1), '.', color='tab:blue')
            axs[3,t].plot(test_true_traj[:,:,T[t], 3].reshape(-1), test_pred_traj[:,:,T[t], 3].reshape(-1), '.', color='tab:blue')
            axs[4,t].plot(test_true_traj[:,:,T[t], 4].reshape(-1), test_pred_traj[:,:,T[t], 4].reshape(-1), '.', color='tab:blue')
            axs[5,t].plot(test_true_traj[:,:,T[t], 5].reshape(-1), test_pred_traj[:,:,T[t], 5].reshape(-1), '.', color='tab:blue')

            axs[5,t].set_xlabel('True')
        axs[2,0].set_ylabel('Prediction')
    
    else:    
        fig, axs = plt.subplots(1,4,figsize=(12,3))
        T = [0, test_true_traj.shape[2]*2//3, test_true_traj.shape[2]//2, -1 ]
        for t in range(len(T)):
            axs[t].plot(test_true_traj[:,:,T[t], 0].reshape(-1), test_pred_traj[:,:,T[t], 0].reshape(-1), '.', color='tab:blue')
            axs[t].set_xlabel('True')
            axs[t].set_ylabel('Prediction')
            axs[t].set_title(f'T = {T[t]}')
    fig.tight_layout() 
    plt.savefig(f"./results/{date}/{model_config}/{args.dynamics}_test_scatter{mlp_idx}.jpg", dpi=1200, format="jpg", bbox_inches='tight')    
main() 
test()