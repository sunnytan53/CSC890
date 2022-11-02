import os
import csv
import glob
import torch
import sys
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if proj_root not in sys.path: sys.path.insert(1, proj_root)

from models.schnet.schnet import SchNetWrap
from data.TorsiAA.TorsiAADataset import TorsiAADataset


def get_best_model(path):
    models = glob.glob(f"{path}/*.pth")
    models = [(float(model.split("/")[-1][4:-4]),model) for model in models]
    models = sorted(models, key=lambda x: x[0])

    print(models)
    return models[0][1]


if __name__ == "__main__":
    dataset = TorsiAADataset(path="../../data/TorsiAA/TorsiAA_initial_geometries/Tyr_mopac_rotations.pickle")
    test_dataloader = DataLoader(dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 collate_fn=dataset.collate,
                                 num_workers=1)

    best = get_best_model("./checkpoints/2022-10-29_18-50-43") 
    print(best)
    model = torch.load(best)
    model.eval()
    print(f"Using model {best}")

    device = "cuda:0"
    for data, label in tqdm(test_dataloader):
        data = {i:v.to(device) for i, v in data.items()}
        label = {i:v.to(device) for i, v in label.items()}
        pred = model(data)
