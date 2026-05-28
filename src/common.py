import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class VAE(nn.Module):
    def __init__(self, n_items, n_lvl2, n_lvl1, n_lvl0):
        super().__init__()
        self.n_items = n_items
        self.n_lvl2 = n_lvl2
        self.n_lvl1 = n_lvl1
        self.n_lvl0 = n_lvl0
        
        self.item_enc = nn.Sequential(
            nn.Linear(n_items, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.lvl2_enc = nn.Sequential(
            nn.Linear(n_lvl2, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.lvl1_enc = nn.Sequential(
            nn.Linear(n_lvl1, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.lvl0_enc = nn.Sequential(
            nn.Linear(n_lvl0, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.fc_mu = nn.Linear(256 + 64 + 32 + 16, 64)
        self.fc_logvar = nn.Linear(256 + 64 + 32 + 16, 64)
        
        self.item_dec = nn.Linear(64, n_items)
        self.lvl2_dec = nn.Linear(64, n_lvl2)
        self.lvl1_dec = nn.Linear(64, n_lvl1)
        self.lvl0_dec = nn.Linear(64, n_lvl0)
    
    def encode(self, x):
        h = torch.cat([
            self.item_enc(x['item']),
            self.lvl2_enc(x['lvl2']),
            self.lvl1_enc(x['lvl1']),
            self.lvl0_enc(x['lvl0'])
        ], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparam(self, mu, logvar, sample=True):
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return {
            'item': self.item_dec(z),
            'lvl2': self.lvl2_dec(z),
            'lvl1': self.lvl1_dec(z),
            'lvl0': self.lvl0_dec(z),
        }
    
    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar, sample=sample)
        recon = self.decode(z)
        return recon, mu, logvar
    
class MultiViewDataset(Dataset):
    def __init__(self, item, lvl2, lvl1, lvl0):
        self.item = item
        self.lvl2 = lvl2
        self.lvl1 = lvl1
        self.lvl0 = lvl0
        
    def __len__(self):
        return len(self.item)
    
    def __getitem__(self, idx):
        return {
            'item': torch.FloatTensor(self.item[idx]),
            'lvl2': torch.FloatTensor(self.lvl2[idx]),
            'lvl1': torch.FloatTensor(self.lvl1[idx]),
            'lvl0': torch.FloatTensor(self.lvl0[idx]),
        }
