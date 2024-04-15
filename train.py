
import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import networkx as nx
import matplotlib as plt

from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv

from gnn.base_gnn import GCN
from dataset import SmallWorldDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model_class, dataset, epochs=10):

    model = model_class().to(device)
    data = dataset[0].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):

        optimizer.zero_grad()

        out, embedding = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}")
        print(f"Training loss: {loss.detach()}")

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

if __name__ == "__main__":
    max_n = 100
    dataset = SmallWorldDataset(None, max_n)
    train(GCN, dataset)
