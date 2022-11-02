import os
import pytz
import torch
import wandb
import shutil
import argparse
import traceback
import matplotlib.pyplot as plt
from tqdm import tqdm
from statistics import mean
from datetime import datetime
import pickle
import sys

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if proj_root not in sys.path: sys.path.insert(1, proj_root)

from models.schnet.schnet import SchNetWrap
from data.MD17.MD17Dataset import MD17SingleDataset
from data.TorsiAA.TorsiAADataset import TorsiAADataset

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------

def EnergyLoss(pred, label):
    p, l = pred["E"].squeeze(), label["E"].squeeze()
    mae = torch.nn.L1Loss()
    return mae(p, l)

def PosForceLoss(pred, label):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    mae = torch.nn.L1Loss()
    return [mae(p[i][j], l[i][j]) for i in range(len(p)) for j in range(3)]

def AtomForceLoss(pred, label):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    p = p.reshape((-1, 3))
    l = l.reshape((-1, 3))
    mae = torch.nn.L1Loss()
    return torch.Tensor([mae(p[i], l[i]) for i in range(len(p))])

def EnergyForceLoss(pred, label):
    E = EnergyLoss(pred, label)
    F = AtomForceLoss(pred, label)
    return E + 30*torch.mean(F)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # dataset = MD17SingleDataset("a",
    #                             "train",
    #                             50000,
    #                             root="../../data/MD17/MD17")

    # wandb.init(project="csc890", entity="bogp")

    dataset = TorsiAADataset(path="../../data/TorsiAA/TorsiAA_initial_geometries/HisE_mopac_rotations.pickle")
                             # path="../../data/TorsiAA/TorsiAA_initial_geometries/Tyr_mopac_rotations.pickle")
                             # path="../../data/TorsiAA/TorsiAA_initial_geometries/Pro_mopac_rotations.pickle")
                             # path="../../data/TorsiAA/TorsiAA_initial_geometries/Thr_mopac_rotations.pickle")
                             # path="../../data/TorsiAA/TorsiAA_initial_geometries/Ala_mopac_rotations.pickle")
    train_dataloader = DataLoader(dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  collate_fn=dataset.collate, num_workers=1)
    
    loss_fn = EnergyForceLoss
    device = "cuda"
    learning_rate = 0.0001
    weight_decay = 1e-5
    save_path = "checkpoints"
    save_freqency = 10

    epochs = 300
    model = SchNetWrap().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                               lr_lambda=lambda epoch: 0.1*(0.9**epoch),
    #                                               last_epoch=-1)

    timezone = pytz.timezone("America/Los_Angeles")
    start_dt = (datetime.now().astimezone(timezone)).strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = f"{save_path}/{start_dt}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    os.mkdir(dir_path)

    for e in tqdm(range(epochs)):
        losses = []
        for data, label in train_dataloader:
            data = {i:v.to(device) for i, v in data.items()}
            label = {i:v.to(device) for i, v in label.items()}
            # print(data, label)

            pred = model(data)
            loss = loss_fn(pred, label)
            
            model.zero_grad()
            loss.backward()
            optimizer.step()

            losses += [loss.to("cpu").item()]
        # scheduler.step()

        epoch_losses = mean(losses)
        if e%(save_freqency) == 0 or e == epochs:
            torch.save(model, f"{dir_path}/{e:03d}_{epoch_losses}.pth")
        print("Average loss", epoch_losses)
# from util import * 

# def setup_dir(args):
#     timezone = pytz.timezone("America/Los_Angeles")
#     start_dt = (datetime.now().astimezone(timezone)).strftime("%Y-%m-%d_%H-%M-%S")
#     dir_path = f"{args.save_path}/{start_dt}"
#     os.mkdir(dir_path)
#     print(f"start traing at {start_dt}")

#     with open(f"{dir_path}/info.txt", 'w') as fp:
#         fp.write(f"model {args.model}\n")
#         fp.write(f"dataset {args.dataset}\n")
#         fp.write(f"split {args.split}\n")
#         fp.write(f"molecule {args.molecule}\n")
#         fp.write(f"loss_fn {args.loss_fn}\n")
#         fp.write(f"optimizer {args.optimizer}\n")
#         fp.write(f"batch_size {args.batch_size}\n")
#         fp.write(f"learning_rate {args.learning_rate}\n")
#         fp.write(f"message {args.message}\n")
#         #fp.write(f" {args.}\n")
    
#     args.start_dt = start_dt
#     args.run_path = dir_path

# def print_info(args):
#     if torch.cuda.is_available():
#         print(f"GPU num detected: {torch.cuda.device_count()}")
#         print(f"Using GPU {torch.cuda.current_device()}")
#         print(torch.cuda.memory_allocated())
#         assert torch.cuda.memory_allocated() < 1000000000
#         args.device = "cuda"
#     else:
#         print(f"Using CPU")
#         args.device = "cpu"

#     print("INFO:")
#     print(f"\tmodel: {args.model}")
#     print(f"\tdataset: {args.dataset}")
#     print(f"\tsplit: {args.split}")
#     print(f"\tmolecule {args.molecule}")
#     print(f"\tloss_fn: {args.loss_fn}")
#     print(f"\toptimizer: {args.optimizer}")
#     print(f"\tbatch_size: {args.batch_size}")
#     print(f"\tlearning_rate: {args.learning_rate}")
#     print("**************************************************************")
#     return

# def wandb_setup(args):
#     wandb.init(project="grad-reg", entity="bogp", group=args.model, name=args.start_dt, id=args.start_dt)
#     wandb.run.summary["model"] = args.model
#     wandb.run.summary["DS/Sp"] = args.dataset+str(args.split)
#     wandb.run.summary["molecule"] = args.molecule
#     wandb.run.summary["loss_fn"] = args.loss_fn
#     return 

# def plot_loss(dir_path, loss):
#     plt.title("Training loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.grid(axis='both')

#     plt.plot(loss, "r-o", label="loss")
#     plt.savefig(f"{dir_path}/loss.png")
#     return 

# def train(args):
#     # initialize
#     args.task = "train"
#     model = get_model(args).to(args.device)
#     dataset = get_dataset(args)
#     train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate, num_workers=16)
#     loss_fn = get_loss_fn(args)
#     optimizer = get_optimizer(model, args)
#     scheduler = get_scheduler(optimizer, args)
    
#     # train
#     hist = {"loss":[]}
#     stop_cnt = 0
#     tq = tqdm(range(args.epoch))
#     for e in tq:
#         # epoch train
#         losses = []
#         for data, label in train_dataloader:
#             data = {i:v.to(args.device) for i, v in data.items()}
#             label = {i:v.to(args.device) for i, v in label.items()}
#             pred = model(data)
#             loss = loss_fn(pred, label)

#             losses.append(loss.to("cpu").item())
#             tq.set_postfix({'mean_loss': mean(losses)})

#             model.zero_grad()
#             loss.backward()
#             optimizer.step()
#         #scheduler.step()

#         epo_loss = mean(losses)

#         # early stopping (basically not using it)
#         if hist["loss"] and abs(epo_loss - hist["loss"][-1]) < 0.01:
#             stop_cnt += 1
#             if stop_cnt == 10:
#                 print(f"early stopping at {e}")
#                 torch.save(model, f"{args.run_path}/{e:03d}_{epo_loss}.pth")
#                 break
#         else: stop_cnt = 0

#         # save model
#         if e%(args.save_freqency) == 0 or e == args.epoch-1:
#             torch.save(model, f"{args.run_path}/{e:03d}_{epo_loss}.pth")

#         # log
#         hist["loss"].append(epo_loss)
#         wandb.log({"loss": epo_loss}, commit=True)
#         #wandb.watch(model)

#     # finish
#     wandb.alert(title=f"Train finish", text=f"Run {args.start_dt} traiing finished")
#     with open(f"{args.run_path}/info.txt", 'a') as fp:
#         losses = hist["loss"]
#         fp.write(f"losses {losses}\n")
#     return

# def main(args):
#     setup_dir(args)
#     try:
#         print_info(args)
#         wandb_setup(args)
#         train(args)
#     except BaseException as e:
#         print(e)
#         traceback.print_exc()
#         ans = input("Ceased. Do you want to delete checkpoint directory? [Y/N]:")
#         if ans in ["Yes", "yes", "Y", "y"]:   
#             wandb.finish()          
#             api = wandb.Api()
#             r = api.run(f"bogp/grad-reg/{args.start_dt}")
#             r.delete()
#             shutil.rmtree(args.run_path)
#             print(f"run {args.start_dt} deleted")

#     return

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
    
#     # env
#     parser.add_argument("--root", type=str, default="../../../datasets/MD17/datas", help="path to the root dir of dataset")
#     parser.add_argument("--save_path", type=str, default="../schnet/checkpoints", help="path to the save dir for trianed checkpoint")
#     parser.add_argument("--message", type=str, default="No message", help="custom message that would be displayed in info.txt")
    
#     # configuration
#     parser.add_argument("-M", "--model", type=str, default="Schnet", help="the model to be trained")
#     parser.add_argument("-D", "--dataset", type=str, default="MD17SingleDataset", help="the dataset to be used")
#     parser.add_argument("-P", "--split", type=int, default=1000, help="the name of dataset subset, aka the number of train samples")
#     parser.add_argument("-m", "--molecule", default="a", type=str, help="lowercase initial of the molecule in the dataset")
#     parser.add_argument("-L", "--loss_fn", type=str, default="EnergyForceLoss", help="the loss fn to be used")
#     parser.add_argument("-O", "--optimizer", type=str, default="Adam", help="the optimizer to be used")
#     parser.add_argument("-S", "--scheduler", type=str, default="LambdaLR", help="the scheduler to be used")

#     # training para
#     parser.add_argument("-e", "--epoch", type=int, default=300, help="number of epoch to train")
#     parser.add_argument("-b", "--batch_size", type=int, default=20, help="batch size to train")
#     parser.add_argument("-s", "--save_freqency", type=int, default=10, help="freqency to save result")
#     parser.add_argument("-l", "--learning_rate", type=float, default=0.0001, help="learning rate to train")
#     parser.add_argument("--weight_decay", type=float, default=0, help="weight decay to train")

#     main(parser.parse_args())