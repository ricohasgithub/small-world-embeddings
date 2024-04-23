
import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import networkx as nx
import matplotlib as plt

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Pad

from gnn.base_gnn import GCN
from gnn.knot_gnn import KnotGCN
from dataset import SmallWorldDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(model_class, dataset, epochs=100):

    train_dataset = dataset[:80]
    test_dataset = dataset[80:]

    train_loader = DataLoader(train_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, shuffle=False)

    model = model_class(2, 10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):

        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            pred, embedding = model(data.to(device))

            loss = F.nll_loss(pred, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach()

        print(f"Epoch: {epoch}")
        print(f"Average training loss: {total_loss/len(train_loader)}")

        model.eval()
        correct = 0
        for data in test_loader:
            pred, embedding = model(data.to(device))
            pred = pred.argmax(dim=1)
            correct += int((pred == data.y).sum())
        print(f"Accuracy: {correct/len(test_loader.dataset):.4f}")

    # Save sample prediction
    for i in range(10):
        pred, embedding = model(train_dataset[i].to(device))
        final_embedding, spherical_embedding = embedding[0].cpu().detach().numpy(), embedding[1].cpu().detach().numpy()
        np.save(f"sample_circle_{i}.npy", spherical_embedding)
        np.save(f"sample_final_{i}.npy", final_embedding)

if __name__ == "__main__":
    max_n = 100
    transform = Pad(max_n)
    dataset = SmallWorldDataset(None, max_n, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=8)
    train(KnotGCN, dataset)
