import copy
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from schedulers import *

def train_vae(model, loader, loss_func, optimizer, cuda=True, epoch=0, kl_warm_up=0, beta=1.):
    model.train()
    #print(model)
    for inputs, _, _ in loader:
        inputs = Variable(inputs).view(inputs.size()[1],-1,inputs.size()[2])
        #print(inputs.size())
        if cuda:
            inputs = inputs.cuda
        output, mean, logvar = model(inputs)
        loss = vae_loss(output, inputs, mean, logvar, loss_func, epoch, kl_warm_up, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, loss

def project_vae(model, loader, cuda=True):
    model.eval()
    print(model)
    for inputs, sample_names, outcomes in loader:
        inputs = Variable(inputs).view(inputs.size()[1],-1,inputs.size()[2])
        if cuda:
            inputs = inputs.cuda()
        z = np.squeeze(model.get_latent_z(inputs).detach().numpy(),axis=1)
    return z, sample_names, outcomes

class AutoEncoder:
    def __init__(self, autoencoder_model, n_epochs, loss_fn, optimizer, cuda=True, kl_warm_up=0,beta=1.,scheduler_opts={}):
        self.model=autoencoder_model
        if cuda:
            self.model = self.model.cuda()
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cuda = cuda
        self.kl_warm_up = kl_warm_up
        self.beta=beta
        self.scheduler = Scheduler(self.optimizer,scheduler_opts) if scheduler_opts else Scheduler(self.optimizer)

    def fit(self, train_data):
        for epoch in range(self.n_epochs):
            model, loss = train_vae(self.model, train_data, self.loss_fn, self.optimizer, self.cuda, epoch, self.kl_warm_up, self.beta)
            self.scheduler.step()
            print("Epoch {}: Loss {}".format(epoch,loss))
        self.model = model
        return model

    def transform(self, train_data):
        return project_vae(self.model, train_data, self.cuda)

    def fit_transform(self, train_data):
        return self.fit(train_data).transform(train_data)

def vae_loss(output, input, mean, logvar, loss_func, epoch, kl_warm_up=0, beta=1.):
    recon_loss = loss_func(output, input)
    if epoch >= kl_warm_up:
        kl_loss = torch.mean(0.5 * torch.sum(
            torch.exp(logvar) + mean**2 - 1. - logvar, 1))
    else:
        kl_loss=0.
    print(recon_loss,kl_loss)
    return recon_loss + kl_loss * beta

class TybaltTitusVAE(nn.Module):
    def __init__(self, n_input, n_latent, hidden_layer_encoder_topology=[100,100,100]):
        super(TybaltTitusVAE, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.pre_latent_topology = [n_input]+(hidden_layer_encoder_topology if hidden_layer_encoder_topology else [])
        self.post_latent_topology = [n_latent]+(hidden_layer_encoder_topology[::-1] if hidden_layer_encoder_topology else [])
        self.encoder_layers = []
        if len(self.pre_latent_topology)>1:
            for i in range(len(self.pre_latent_topology)-1):
                layer = nn.Linear(self.pre_latent_topology[i],self.pre_latent_topology[i+1])
                torch.nn.init.xavier_uniform(layer.weight)
                self.encoder_layers.append(nn.Sequential(layer,nn.ReLU()))
        self.encoder = nn.Sequential(*self.encoder_layers) if self.encoder_layers else nn.Dropout(p=0.)
        self.z_mean = nn.Linear(self.pre_latent_topology[-1],n_latent)
        self.z_var = nn.Linear(self.pre_latent_topology[-1],n_latent)
        self.z_develop = nn.Linear(n_latent,self.pre_latent_topology[-1])
        self.decoder_layers = []
        if len(self.post_latent_topology)>1:
            for i in range(len(self.post_latent_topology)-1):
                layer = nn.Linear(self.post_latent_topology[i],self.post_latent_topology[i+1])
                torch.nn.init.xavier_uniform(layer.weight)
                self.decoder_layers.append(nn.Sequential(layer,nn.ReLU()))
        self.decoder_layers = nn.Sequential(*self.decoder_layers)
        self.output_layer = nn.Sequential(nn.Linear(self.post_latent_topology[-1],n_input),nn.Sigmoid())
        if self.decoder_layers:
            self.decoder = nn.Sequential(*[self.decoder_layers,self.output_layer])
        else:
            self.decoder = self.output_layer

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = Variable(torch.randn(stddev.size()))
        return (noise * stddev) + mean

    def encode(self, x):
        x = self.encoder(x)
        #print(x.size())
        #x = x.view(x.size(0), -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        #print('mean',mean.size())
        return mean, var

    def decode(self, z):
        #out = self.z_develop(z)
        #print('out',out.size())
        #out = out.view(z.size(0), 64, self.z_dim, self.z_dim)
        out = self.decoder(z)
        #print(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar

    def get_latent_z(self, x):
        mean, logvar = self.encode(x)
        return self.sample_z(mean, logvar)

class CVAE(nn.Module):
    def __init__(self, in_shape, n_latent):
        super(CVAE,self).__init__()
        self.in_shape = in_shape
        self.n_latent = n_latent
        c,h,w = in_shape
        self.z_dim = h//2**2 # receptive field downsampled 2 times
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),  # 32, 16, 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32, 8, 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.z_mean = nn.Linear(64 * self.z_dim**2, n_latent)
        self.z_var = nn.Linear(64 * self.z_dim**2, n_latent)
        self.z_develop = nn.Linear(n_latent, 64 * self.z_dim**2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1),
            CenterCrop(h,w),
            nn.Sigmoid()
        )

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = Variable(torch.randn(stddev.size()))
        return (noise * stddev) + mean

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    def decode(self, z):
        out = self.z_develop(z)
        out = out.view(z.size(0), 64, self.z_dim, self.z_dim)
        out = self.decoder(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar

    def get_latent_z(self, x):
        mean, logvar = self.encode(x)
        return self.sample_z(mean, logvar)

def train_classify(model, loader, loss_func, optimizer, cuda=True, epoch=0):
    model.train()
    #print(model)
    for inputs, _, _ in loader: # change dataloder for classification/regression tasks
        inputs = Variable(inputs).view(inputs.size()[1],-1,inputs.size()[2])
        #print(inputs.size())
        if cuda:
            inputs = inputs.cuda
        output = model(inputs)
        loss = loss_func(output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, loss

class VAE_Classifier:
    pass

class VAE_MLP(nn.Module):
    def __init__(self, vae_model, n_output, hidden_layer_topology=[100,100,100]):
        super(VAE_MLP,self).__init__()
        self.vae = vae_model.vae_model
        self.n_latent = vae_model.n_latent
        self.n_output = n_output
        self.topology = [self.n_latent]+(hidden_layer_topology if hidden_layer_topology else [])
        self.mlp_layers = []
        if len(self.topology)>1:
            for i in range(len(self.topology)-1):
                layer = nn.Linear(self.topology[i],self.topology[i+1])
                torch.nn.init.xavier_uniform(layer.weight)
                self.mlp_layers.append(nn.Sequential(layer,nn.ReLU()))
        self.output_layer = nn.Linear(self.topology[-1],self.n_output)
        torch.nn.init.xavier_uniform(self.output_layer.weight)
        self.mlp_layers.extend([self.output_layer,nn.Sigmoid()])
        self.mlp = self.Sequential(*self.mlp_layers)

    def forward(self,x):
        out=self.vae.get_latent_z(x)
        return self.mlp(out)
