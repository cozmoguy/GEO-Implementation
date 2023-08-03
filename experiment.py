import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from helpers import multiDistribution, generate_random_noise
from problem_objective import objective
from models import Generator, Discriminator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

initial_seed = torch.from_numpy(np.array(pd.read_csv("initial_seed.csv", index_col=0))).to(device)

sample_dim = 5
batch_size = 256
num_epochs = 200
learning_rate = 0.0002

generator = Generator(sample_dim, sample_dim).to(device)
discriminator = Discriminator(sample_dim).to(device)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

distri = multiDistribution(initial_seed)

pbar = tqdm(range(num_epochs))
new_gens = np.array([])
initial_seed_obj = objective(initial_seed)
ground_val = initial_seed_obj.min().item()
for epoch in pbar:
    if new_gens.any():
        initial_seed = torch.cat((initial_seed, new_gens),dim=0)
        initial_seed_obj = objective(initial_seed)
        sort_ind = torch.sort(initial_seed_obj,dim=0)[1]
        initial_seed = initial_seed[sort_ind]
        initial_seed = initial_seed[:5000]
        ground_val = initial_seed_obj.min().item()

        distri = multiDistribution(initial_seed)
        new_gens = np.array([])
    mean, std = torch.mean(initial_seed_obj), torch.std(initial_seed_obj)
    for batch_idx in range(10000 // batch_size):
        # Train the Discriminator
        discriminator.zero_grad()

        real_samples = distri.sample(sample_shape = [batch_size]).to(device).float()
        real_samples /= real_samples.sum(dim=1, keepdim=True)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_samples = generator(generate_random_noise(batch_size, 5)).detach()
        fake_labels = torch.zeros(batch_size, 1).to(device)

        d_real_output = discriminator(real_samples)
        d_real_loss = criterion(d_real_output, real_labels)

        d_fake_output = discriminator(fake_samples)
        d_fake_loss = criterion(d_fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train the Generator
        generator.zero_grad()

        generated_samples = generator(generate_random_noise(batch_size, 5))
        g_fake_output = discriminator(generated_samples)
        g_loss = criterion(g_fake_output, real_labels)

        g_loss.backward()
        g_optimizer.step()
    new_gens = generator(generate_random_noise(1000, 5)).detach()
    new_gens /= new_gens.sum(dim=1, keepdim=True)
    pbar.set_description(f"{round(g_loss.item(),2)} :: {round(d_loss.item(),2)} :: {ground_val} :: {mean.item(), std.item()}")