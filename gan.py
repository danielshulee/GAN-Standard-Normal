"""
Name: gan.py
Author: Daniel S. Lee
Description: Loads all pytorch functions and packages needed to train 4 GANs
(original, nonsaturating, Wasserstein, gradient penalty) to sample from a 1D 
standard normal distribution. Imports code for:

* Data: functions to sample from the noise and true distribution
* Model: fully connected network with initialization
* Loss: implements generator and discriminator losses for original,
nonsaturating, Wasserstein, gradient penalty GAN
* Training and visualization: functions to train a GAN and visualize the final
generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
import scipy.stats as stats
from functools import partial
from statsmodels.distributions.empirical_distribution import ECDF
import progressbar

## DATA
def norm_pdf(x): return stats.norm.pdf(x)
def norm_cdf(x): return stats.norm.cdf(x)
def sample_uniform(bs, **kwargs):
  """
  Returns a uniform sample of size bs.
  """
  sample = np.random.uniform(size=int(bs), **kwargs)
  return torch.tensor(sample).reshape((-1,1)).float()
def sample_gaussian(bs, **kwargs):
  """
  Returns a gaussian sample of size bs.
  """
  sample = np.random.normal(size=int(bs), **kwargs)
  return torch.tensor(sample).reshape((-1,1)).float()

## MODEL
def init_weights(m, bias=True, **kwargs):
  """
  Initializes a linear layer with Xavier initialization.
  """
  if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight, gain=5/3) # gain=5/3 is recommended for Tanh
      if bias==True:m.bias.data.fill_(0.01)
def gan_model(layer_sizes=[16,16,16], **kwargs):
  """
  Given a list of hidden layer sizes, creates a fully connected network
  with tanh activation functions. Initializes with Xavier initialization.
  """
  layer_sizes = [1] + layer_sizes + [1]
  layers = []
  for i in range(len(layer_sizes)-1):
    layers.append(nn.Linear(*layer_sizes[i:i+2],**kwargs))
    layers.append(nn.Tanh())
  return nn.Sequential(*layers[:-1]).apply(partial(init_weights,**kwargs)) # remove last sigmoid
def save_gan(g_model, name, d_model=None, save_d=False):
  """
  Saves a GAN (optionally critic) to Google Drive with a name.
  """
  g_path = "/content/drive/MyDrive/" + name + "-generator.pt"
  d_path = "/content/drive/MyDrive/" + name + "-critic.pt"
  torch.save(g_model.state_dict(), g_path)
  if save_d:
    torch.save(d_model.state_dict(), d_path)
def load_gan(name, model_function=gan_model, load_d=False):
  """
  Loads and returns a GAN (optionally critic) from Google Drive with name.
  """
  generator = model_function()
  generator.load_state_dict(torch.load("/content/drive/MyDrive/" + name + "-generator.pt"))
  if load_d:
    critic = model_function()
    critic.load_state_dict(torch.load("/content/drive/MyDrive/a" + name + "-critic.pt"))
    return (generator, critic)
  return generator

## LOSS
# Standard disciminator loss
critic_loss = F.binary_cross_entropy_with_logits
# Non-saturating and saturating generator loss
def unsaturating_gen_loss(critic_preds):
  """
  Altnerate generator loss from the original paper, which ascends
  log(D(G(Z))).
  
  This is equivalent to descending the binary cross entropy loss of the
  critic with the wrong label (1).
  """
  return F.binary_cross_entropy_with_logits(input=critic_preds, target=torch.ones_like(critic_preds))
def vanilla_gen_loss(critic_preds):
  """
  Original (saturating) generator loss, which descends log(1-D(G(z))). There
  should be no worries of numerical instability, unless the discriminator guesses
  a value overly close to 1 (unlikely; D(G(z))=1 --> log(1-1)=undefined)
  """
  return torch.log(1-critic_preds.sigmoid()).mean()
# WGAN loss
def wgan_critic_loss(input, target):
  """
  WGAN critic loss, which ascends E[D(x)] - E[D(G(z))]. In my implementation,
  D(x) is the upper half of input and D(G(z)) is the lower (second) half. Labels
  (0s and 1s) are disregarded as the WGAN critic no longer acts as a binary
  classifier.
  """
  middle_idx = int(input.shape[0]/2)
  real_pred = input[:middle_idx]
  fake_pred = input[middle_idx:]
  return fake_pred.mean() - real_pred.mean()
def wgan_gen_loss(critic_preds):
  """
  WGAN generator loss, which descends the Wasserstein distance:
  d[ E[D(x)] - E[D(G(z))] ]/dG = -d[ E[D(G(z))] ]/dG
  """
  return -critic_preds.mean()
# Gradient penalty: uses the same discriminator loss and adds a regularizer term
# (wrapped within training function). The unsaturation generator loss is used.

## TRAINING
def train_d_gp(opt, g_model, d_model, d_loss_func, bs, gp, clip, sample_noise, sample_true, device):
  """
  Given a generative and discriminative model, loss function, and optimizer, trains the discriminator for one batch.
  Optionally adds zero centered gradient penalty, controlled by the gp argument (set to 0 to disable). If clip is
  set to a float, implements gradient clipping for WGAN.

  Returns the loss.
  """
  # Clip discriminator weights is clip is a float
  if clip:
    assert isinstance(clip, (float,int))
    for p in d_model.parameters(): p.data.clamp_(-clip, clip)
  # Generate batch of (data, labels)
  xb_real = sample_true(bs/2).requires_grad_().to(device)
  yb_real = torch.ones_like(xb_real)
  xb_fake = g_model(sample_noise(bs/2).to(device))
  yb_fake = torch.zeros_like(xb_real)
  xb = torch.cat((xb_real, xb_fake))
  yb = torch.cat((yb_real, yb_fake)).to(device)
  # Predict on data to yield (D(data), labels)
  preds = d_model(xb)
  # Compute loss: L(D(data), labels), then add gradient penalty.
  loss = d_loss_func(preds, yb)
  real_preds = preds[:int(bs/2)].sigmoid()
  # Compute independently for each true sample the derivative of the critic confidence
  # with respect to the input
  if gp!=0:
    d_grad = torch.autograd.grad(real_preds, xb_real, torch.ones_like(real_preds), create_graph=True)[0]
    loss += gp * torch.mean(torch.square(d_grad))
  # Compute gradients
  opt.zero_grad()
  g_model.zero_grad()
  loss.backward()
  # Step
  opt.step()
  return loss

def train_g(opt, g_model, d_model, g_loss_func, bs, sample_noise, device):
  """
  Given a generative and discriminative model, loss function, and optimizer, trains the generator for one batch.

  Returns the loss.
  """
  # Generate batch of sample samples G(z)
  zb = sample_noise(bs).to(device)
  xb = g_model(zb).to(device)
  # Predict on data with critic to yield D(G(z))
  preds = d_model(xb)
  # Compute loss: L(D(G(z)))
  loss = g_loss_func(preds)
  # Compute gradients
  opt.zero_grad()
  d_model.zero_grad()
  loss.backward()
  # Step
  opt.step()
  return loss

def train_gan_gp(g_model, d_model, g_loss_func=vanilla_gen_loss, d_loss_func=critic_loss, \
  opt_func=partial(torch.optim.Adam, betas=(0,0.9)), n_steps=100, g_steps=1, d_steps=5, \
    lr=1e-5, bs=256, gp=10, clip=False, compute_KS_every=10, \
    sample_noise=sample_uniform, sample_true=sample_gaussian, cdf_true=norm_cdf):
  """
  Trains a GAN model for n steps, which each step training the generator for `g_steps` turns
  (batches) and the disciminator for `d_steps` turns.

  Returns four items: a tensor of generator losses for every epoch, generator samples,
  discriminator losses, and the KS-test value for every 
  """
  # Setup
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  g_model.to(device)
  d_model.to(device)
  g_opt = opt_func(g_model.parameters(), lr=lr)
  d_opt = opt_func(d_model.parameters(), lr=lr)
  g_losses = []
  g_samples = []
  d_losses = []
  ks_values = []
  # Top training loop:
  for i in progressbar.progressbar(range(n_steps)):
    # Train generator first
    g_loss = 0
    for j in range(g_steps):
      loss = train_g(g_opt, g_model, d_model, g_loss_func, bs, sample_noise, device)
      g_loss += loss/g_steps
    g_losses.append(g_loss)
    # See some generated batches
    g_samples.append(g_model(sample_noise(bs*8).to(device)).squeeze().detach())
    # Train disciminator
    d_loss = 0
    for k in range(d_steps):
      loss = train_d_gp(d_opt, g_model, d_model, d_loss_func, bs, gp, clip, sample_noise, sample_true, device)
      d_loss += loss/d_steps
    d_losses.append(d_loss)
    # Compute KS-Test value
    if i%compute_KS_every == 0:
      samples = g_model(sample_noise(10000).to(device)).squeeze().detach().cpu().numpy()
      ks = stats.kstest(samples, cdf_true).statistic
      ks_values.append(ks)
  return torch.stack(g_losses).detach().cpu(), torch.stack(g_samples).cpu(), torch.stack(d_losses).detach().cpu(), torch.tensor(ks_values).float()

## VISUALIZATION

def visualize_training(training_results, iterations, pdf_func=norm_pdf, xlim=(-3,3)):
  """
  Visualize training dynamics via generator/discriminator losses and generated samples.
  """
  # Setup
  g_losses, g_samples, d_losses = training_results
  x = np.linspace(*xlim,100)
  n_steps = len(d_losses)
  n = min(iterations, n_steps)
  # Create figure
  fig, ax = plt.subplots(1,2,figsize=(8,4))
  ax[0].set_title("Generator Samples")
  ax[1].set_title("Losses")
  ax[0].set_xlim(*xlim)
  ax[1].set_xlim(0, n_steps)
  # Left plot: generator samples
  ax[0].plot(x, pdf_func(x))
  ax[0].hist(g_samples[max(0,n-10):n+10].view(-1), bins=25, density=True, alpha=0.75)
  # Right plot: loss
  ax[1].plot(g_losses[:n], label="Generator loss", linewidth=0.1)
  ax[1].plot(d_losses[:n], label="Critic Loss", linewidth=0.1)
  ax[1].legend()
  # Main title
  fig.suptitle(f"Epoch {n}/{n_steps}")

def visualize_gan(g_model, name, ks_values, ks_interval=10, sample_noise=sample_uniform, pdf_func=norm_pdf, \
                  cdf_func=stats.norm.cdf, n=1e7, xlim=(-4,4), bins=50):
  """
  Visualizes the final generator. Plots the empirical PDF and CDF generated from n samples with the 
  true distribution. Also plots KS-Test Values over time.
  """
  # Generate 1D numpy array of generated samples, setup for for plotting
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  samples = g_model(sample_noise(n).to(device)).squeeze().detach().cpu().numpy()
  fig, ax = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
  x = np.linspace(*xlim, 100)
  ax[0].set_title(f"PDF comparison")
  ax[1].set_title(f"CDF comparison")
  # Plot true/empirical PDF (left), CDF (right)
  ax[0].plot(x,pdf_func(x), label="True PDF")
  ax[0].hist(samples, bins=bins, density=True, alpha=0.75, label="Generator samples")
  ax[0].legend(loc="upper left")
  ecdf = ECDF(samples)
  ax[1].plot(x, cdf_func(x), label="True CDF")
  ax[1].plot(x, ecdf(x), label="Empirical CDF")
  ax[1].legend(loc="upper left")
  ax[0].set_xlabel("Sample space")
  ax[1].set_xlabel("Sample space")
  # Plot KS-Test over time
  num_ks_values = ks_values.shape[0]
  ax[2].plot(np.arange(0,num_ks_values*ks_interval,ks_interval), ks_values.numpy(), linewidth=0.5)
  ax[2].set_title("KS Statistic Throughout Training")
  ax[2].set_xlabel("Epoch")
  # Set figure title
  final_ks = stats.kstest(samples, cdf_func).statistic
  fig.suptitle(f"{name}:\nKS Statistic: {round(final_ks,4)}",fontsize=14)
  return samples