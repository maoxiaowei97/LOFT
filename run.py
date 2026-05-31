import argparse
import torch
import os
import time
import configparser
from dataset_traffic import get_dataloader
from main_model import LOFT_Traffic
from utils import train, evaluate, set_seed
import logging  
import sys

parser = argparse.ArgumentParser(description="LOFT Time Series Imputation")

parser.add_argument("--config", type=str, default="config/PEMS04.conf")
parser.add_argument("--dataset", type=str, default="PEMS04")
parser.add_argument("--miss_type", type=str, default=None)
parser.add_argument('--miss_rate',type=str,default=None)
parser.add_argument('--device', default=None)
parser.add_argument('--mode', type=str, default='eval', choices=['train', 'eval'], help='train or eva;')
parser.add_argument("--seed", type=int, default=126)
parser.add_argument("--alpha_warmup_ratio", type=str, default=None, help="Ratio of epochs for alpha transition")

parser.add_argument("--initial_lr", type=float, default=None)
parser.add_argument("--final_lr", type=float, default=None)
parser.add_argument("--high_lr_epochs", type=int, default=None)

parser.add_argument("--savename",type=str,default="")
parser.add_argument("--results_file", type=str, default=None)
parser.add_argument("--cond_path", type=str, default="params/PEMS04_SC-TC_0.8_20260215_125240_cond.pth")
parser.add_argument("--logfile", type=str, default="")

parser.add_argument(
    "--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"]
)

parser.add_argument("--lr", type=float, default=None, help="Override lr in config")
parser.add_argument("--num_steps", type=int, default=None, help="Override num_steps in config")
parser.add_argument("--inference_steps", type=int, default=None, help="Override inference_steps in config")
parser.add_argument("--epochs", type=int, default=None, help="Override epochs in config")
parser.add_argument("--min_alpha", type=float, default=None)
parser.add_argument("--exp_for_hard", type=float, default=None)
parser.add_argument("--exp_for_easy", type=float, default=None)

args = parser.parse_args()

set_seed(args.seed)


if args.dataset:
    dataset_name = args.dataset
    args.config = f"config/{dataset_name}.conf"
else:
    dataset_name = os.path.splitext(os.path.basename(args.config))[0]

config = configparser.ConfigParser()
config.read(args.config)

if args.initial_lr is not None:
    config["train"]["initial_lr"] = str(args.initial_lr)

if args.final_lr is not None:
    config["train"]["final_lr"] = str(args.final_lr)

if args.high_lr_epochs is not None:
    config["train"]["high_lr_epochs"] = str(args.high_lr_epochs)

if args.miss_type is not None:
    config["train"]["type"] = args.miss_type
if args.miss_rate is not None:
    config["train"]["miss_rate"] = args.miss_rate

if args.alpha_warmup_ratio is not None:
    config["train"]["alpha_warmup_ratio"] = str(args.alpha_warmup_ratio)

config["model"]["target_strategy"] = args.targetstrategy

if args.lr is not None:
    config["train"]["lr"] = str(args.lr)
if args.num_steps is not None:
    config["flow_matching"]["num_steps"] = str(args.num_steps)
if args.inference_steps is not None:
    config["flow_matching"]["inference_steps"] = str(args.inference_steps)
if args.epochs is not None:
    config["train"]["epochs"] = str(args.epochs)

if args.min_alpha is not None:
    config["flow_matching"]["min_alpha"] = str(args.min_alpha)

if args.exp_for_hard is not None:
    config["flow_matching"]["exp_for_hard"] = str(args.exp_for_hard)

if args.exp_for_easy is not None:
    config["flow_matching"]["exp_for_easy"] = str(args.exp_for_easy)

if args.device is not None:
    config["model"]["device"] = args.device
else:
    if "model" in config and "device" in config["model"]:
        args.device = config["model"]["device"]
    else:
        args.device = "cuda:0"
        if "model" not in config: config["model"] = {}
        config["model"]["device"] = args.device

miss_type = config['train']['type']
miss_rate = config['train']['miss_rate']

current_time_str = time.strftime("%Y%m%d_%H%M%S")

if not args.logfile:
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    args.logfile = f"./logs/log_{dataset_name}_{current_time_str}.log"

if not args.results_file:
    if not os.path.exists("./results"):
        os.makedirs("./results")
    args.results_file = f"./results/results_{dataset_name}_{current_time_str}.csv"

if not args.savename:
    args.savename = f"{dataset_name}_{miss_type}_{miss_rate}_{current_time_str}"

if args.logfile:
    log_dir = os.path.dirname(args.logfile)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(args.logfile, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

base_name = f"{dataset_name}_{miss_type}_{miss_rate}"
savename_to_use = args.savename
results_file_to_use = args.results_file

logging.info("\n" + "="*50)
logging.info("="*50)
logging.info(f"mode: {args.mode}")
logging.info(f"dataset_name: {dataset_name} (从 '{args.config}')")
logging.info(f"miss_type: {miss_type}")
logging.info(f"miss_rate: {miss_rate}")
logging.info(f"epoch: {config['train']['epochs']}")
logging.info(f"savename: {savename_to_use}")
logging.info(f"results_file: {results_file_to_use}")
logging.info(f"device: {args.device}")
logging.info(f"seed: {args.seed}")
logging.info(f"device: {config['model']['device']}")
logging.info(f"lr: {config['train']['lr']}")
logging.info(f"num_steps: {config['flow_matching']['num_steps']}")
logging.info(f"inference_steps: {config['flow_matching']['inference_steps']}")
logging.info(f"alpha_warmup_ratio: {config['train']['alpha_warmup_ratio']}")
logging.info(f"high_lr_epochs: {config['train']['high_lr_epochs']}")
logging.info(f"layers: {config['flow_matching']['layers']}")
logging.info(f"min alpha: {config['flow_matching']['min_alpha']}")

use_nni = int(config['train']['use_nni'])

data_prefix = config['file']['data_prefix']
if not os.path.isabs(data_prefix):
    data_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_prefix)

val_ratio = float(config['train']['val_ratio'])
test_ratio = float(config['train']['test_ratio'])
sample_len = int (config['train']['sample_len'])
batch_size = int (config['train']['batch_size'])
nsample = int(config['flow_matching']['nsample'])

true_datapath = os.path.join(data_prefix,f"true_data_{miss_type}_{miss_rate}_v2.npz")
miss_datapath = os.path.join(data_prefix,f"miss_data_{miss_type}_{miss_rate}_v2.npz")

imputed_data_dir = config['file']['imputed_data_dir']
if not os.path.isabs(imputed_data_dir):
    imputed_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), imputed_data_dir)

imputed_datapath = os.path.join(imputed_data_dir, f"imputed_{dataset_name}_{miss_type}_{miss_rate}.npz")

train_loader, valid_loader, test_loader, target_dim, _std, _mean = get_dataloader(
    true_datapath,miss_datapath,imputed_datapath,val_ratio,test_ratio,batch_size,sample_len
)
model = LOFT_Traffic(config, target_dim, args.device).to(args.device)


if args.mode == 'train':
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        _std=_std,
        _mean=_mean,
        savename = savename_to_use
    )

    tensor_save_name = f"evaluation_tensors_{dataset_name}_{miss_type}_{miss_rate}_{current_time_str}.pth"
    tensor_save_path = os.path.join("./results", tensor_save_name)
    
    evaluate(
        model,
        test_loader,
        _std, _mean, use_nni,
        nsample=nsample,
        results_file = results_file_to_use,
        tensor_save_path=tensor_save_path
    )

elif args.mode == 'eval':

    if args.cond_path:
        if not os.path.isabs(args.cond_path):
             args.cond_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.cond_path)

        model.velocity_net.load_state_dict(torch.load(args.cond_path,map_location = args.device) )
    
    model.eval()

    tensor_save_name = f"evaluation_tensors_{dataset_name}_{miss_type}_{miss_rate}_{current_time_str}.pth"
    tensor_save_path = os.path.join("./results", tensor_save_name)
    
    evaluate(
        model,
        test_loader,
        _std, _mean, use_nni,
        nsample=nsample,
        results_file=results_file_to_use,
        tensor_save_path=tensor_save_path
    )