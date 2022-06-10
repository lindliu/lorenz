#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:11:36 2022

@author: do0236li
"""

import logging
import os
from typing import Sequence

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal

import torchsde

class StochasticLorenz(object):
    """Stochastic Lorenz attractor.
    Used for simulating ground truth and obtaining noisy data.
    Details described in Section 7.2 https://arxiv.org/pdf/2001.01328.pdf
    Default a, b from https://openreview.net/pdf?id=HkzRQhR9YX
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (.1, .28, .3)):
        super(StochasticLorenz, self).__init__()
        self.a = a
        self.b = b

    def f(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        a1, a2, a3 = self.a

        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3
        return torch.cat([f1, f2, f3], dim=1)

    def g(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        b1, b2, b3 = self.b

        g1 = x1 * b1
        g2 = x2 * b2
        g3 = x3 * b3
        return torch.cat([g1, g2, g3], dim=1)

    @torch.no_grad()
    def sample(self, x0, ts, noise_std, normalize):
        """Sample data for training. Store data normalization constants if necessary."""
        xs = torchsde.sdeint(self, x0, ts)
        if normalize:
            mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
            xs.sub_(mean).div_(std).add_(torch.randn_like(xs) * noise_std)
        return xs


def make_dataset(t0, t1, batch_size, noise_std, train_dir, device):
    data_path = os.path.join(train_dir, 'lorenz_data_f.pth')

    _y0 = torch.randn(batch_size, 3, device=device)
    ts = torch.linspace(t0, t1, steps=100, device=device)
    xs = StochasticLorenz().sample(_y0, ts, noise_std, normalize=True)

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    torch.save({'xs': xs, 'ts': ts}, data_path)
    logging.warning(f'Stored toy data at: {data_path}')
    return xs, ts


class model_f_lorenz(nn.Module):
    def __init__(self):
        super(model_f_lorenz, self).__init__()
        self.l1 = nn.Linear(3,64)
        self.l2 = nn.Linear(64,128)
        self.l3 = nn.Linear(128,128)
        self.l4 = nn.Linear(128,64)
        self.l5 = nn.Linear(64,3)
        self.rl = nn.Tanh()
        
    def forward(self, x):
        x = self.rl(self.l1(x))
        x = self.rl(self.l2(x))
        x = self.rl(self.l3(x))
        x = self.rl(self.l4(x))
        x = self.l5(x)
        return x
    
    
if __name__ == "__main__":

    batch_size=1024
    latent_size=4
    context_size=64
    hidden_size=128
    lr_init=1e-2
    t0=0.
    t1=2.
    lr_gamma=0.997
    num_iters=5000
    kl_anneal_iters=1000
    pause_every=50
    noise_std=0.01
    adjoint=False
    train_dir='./dump/lorenz/'
    method="euler"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xs, ts = make_dataset(t0=t0, t1=t1, batch_size=batch_size*100, noise_std=noise_std, train_dir=train_dir, device=device)
    xs = xs.reshape([-1,3])
    k=2
    dx_t = (xs[k:,:]-xs[:-k,:])/(ts[k]-ts[0])
    # plt.scatter(xs[:-k,0][::10],xs[:-k,1][::10],c=dx_t[:,0][::10],s=.1)
    
    model_f = model_f_lorenz().to(device)
    
    
    inputs = xs[:-k,:].clone().to(device)
    target = dx_t.clone().to(device)
    ## train model by data
    mse = torch.nn.MSELoss() # Mean squared error
    optim_F = optim.Adam(model_f.parameters(), lr=1e-3)
    
    batchsize = 1024
    for i in range(50000):
        idx = np.random.randint(0,inputs.shape[0],batchsize)
        # t_in = torch.linspace(0,1,num).to(device)
        # t_in = t_in[idx].reshape(batchsize,1)
        # t_in.requires_grad = True
        
        txs = inputs[idx]
        ###directly learn ode, N=x'###
        u_out = model_f(txs)
        target_f = target[idx]
        loss = torch.mean(mse(u_out, target_f))
            

        loss.backward(retain_graph=True)
        optim_F.step()
        optim_F.zero_grad()
        
        if i%1000==0:
            print('iter:{}. derivative loss: {:.6f}'.format(i, loss.item()))
            
    model_f_path = os.path.join(train_dir, 'model_f.pt')
    torch.save(model_f.state_dict(), model_f_path)
    
    
    pred = model_f(inputs[::100,:]).detach()
    vmin = min(pred.min().item(), dx_t.min().item())
    vmax = max(pred.max().item(), dx_t.max().item())
    fig, axes = plt.subplots(1,2,figsize=[10,3])
    axes[0].scatter(xs[:-k,0][::100],xs[:-k,1][::100],c=dx_t[:,0][::100],s=.1, vmin=vmin, vmax=vmax)
    axes[1].scatter(xs[:-k,0][::100],xs[:-k,1][::100],c=pred[:,0],s=.1, vmin=vmin, vmax=vmax)
    