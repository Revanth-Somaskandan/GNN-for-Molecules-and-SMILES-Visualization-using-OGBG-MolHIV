import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx

from ogb.graphproppred import PygGraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Draw
dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

split_idx = dataset.get_idx_split()

train_dataset = dataset[split_idx["train"]]
valid_dataset = dataset[split_idx["valid"]]
test_dataset = dataset[split_idx["test"]]

print("Train graphs:", len(train_dataset))
print("Validation graphs:", len(valid_dataset))
print("Test graphs:", len(test_dataset))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
class GNN(torch.nn.Module):

    def __init__(self, num_features):
        super(GNN, self).__init__()

        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 64)

        self.linear = torch.nn.Linear(64, 1)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = x.float()
        edge_index = edge_index.long()

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.linear(x)

        return x
model = GNN(dataset.num_node_features)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = torch.nn.BCEWithLogitsLoss()
epochs = 20
loss_history = []

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for data in train_loader:

        optimizer.zero_grad()

        pred = model(data)

        target = data.y.float()

        loss = loss_fn(pred, target)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    plt.figure()

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")

plt.show()
sample = dataset[0]

G = to_networkx(sample, to_undirected=True)

plt.figure(figsize=(6,6))

nx.draw(G, node_color="skyblue", with_labels=True)

plt.title("Molecular Graph Structure")

plt.show()
import pandas as pd
smiles_df = pd.read_csv("dataset/ogbg_molhiv/mapping/mol.csv.gz")

smiles = smiles_df.iloc[0]["smiles"]

print("SMILES:", smiles)
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

mol = Chem.MolFromSmiles(smiles)

img = Draw.MolToImage(mol)

plt.imshow(img)
plt.axis("off")
plt.title("Molecule from SMILES")
plt.show()