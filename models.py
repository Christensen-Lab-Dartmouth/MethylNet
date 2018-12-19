from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from schedulers import *
from plotter import *
from sklearn.preprocessing import LabelEncoder
from visualizations import umap_embed, plotly_plot
import copy

def train_vae(model, loader, loss_func, optimizer, cuda=True, epoch=0, kl_warm_up=0, beta=1.):
    model.train(True) #FIXME
    #print(model)
    total_loss,total_recon_loss,total_kl_loss=0.,0.,0.
    for inputs, _, _ in loader:
        inputs = Variable(inputs).view(inputs.size()[0],inputs.size()[1]) # modify for convolutions, add batchnorm2d?
        #print(inputs.size())
        if cuda:
            inputs = inputs.cuda()
        output, mean, logvar = model(inputs)
        loss, reconstruction_loss, kl_loss = vae_loss(output, inputs, mean, logvar, loss_func, epoch, kl_warm_up, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        total_recon_loss+=reconstruction_loss.item()
        total_kl_loss+=kl_loss.item()
    return model, total_loss,total_recon_loss,total_kl_loss

def project_vae(model, loader, cuda=True):
    model.eval()
    #print(model)
    final_outputs=[]
    sample_names_final=[]
    outcomes_final=[]
    for inputs, sample_names, outcomes in loader:
        inputs = Variable(inputs).view(inputs.size()[0],inputs.size()[1]) # modify for convolutions, add batchnorm2d?
        if cuda:
            inputs = inputs.cuda()
        z = np.squeeze(model.get_latent_z(inputs).detach().cpu().numpy())
        final_outputs.append(z)
        sample_names_final.extend([name[0] for name in sample_names])
        outcomes_final.extend([outcome[0] for outcome in outcomes])
    z=np.vstack(final_outputs)
    sample_names=np.array(sample_names_final)
    outcomes=np.array(outcomes_final)
    return z, sample_names, outcomes

class AutoEncoder:
    def __init__(self, autoencoder_model, n_epochs, loss_fn, optimizer, cuda=True, kl_warm_up=0,beta=1.,scheduler_opts={}):
        self.model=autoencoder_model
        print(self.model)
        if cuda:
            self.model = self.model.cuda()
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cuda = cuda
        self.kl_warm_up = kl_warm_up
        self.beta=beta
        self.scheduler = Scheduler(self.optimizer,scheduler_opts) if scheduler_opts else Scheduler(self.optimizer)
        self.vae_animation_fname='animation.mp4'
        self.loss_plt_fname='loss.png'
        self.plot_interval=5
        self.embed_interval=200
        self.validation_set = False


    def fit(self, train_data):
        loss_list = []
        best_model=None
        animation_plts=[]
        plt_data={'kl_loss':[],'recon_loss':[],'lr':[]}
        for epoch in range(self.n_epochs):
            model, loss, recon_loss, kl_loss = train_vae(self.model, train_data, self.loss_fn, self.optimizer, self.cuda, epoch, self.kl_warm_up, self.beta)
            self.scheduler.step()
            plt_data['kl_loss'].append(kl_loss)
            plt_data['recon_loss'].append(recon_loss)
            plt_data['lr'].append(self.scheduler.get_lr())
            print("Epoch {}: Loss {}, Recon Loss {}, KL-Loss {}".format(epoch,loss,recon_loss,kl_loss))
            if epoch > self.kl_warm_up:
                loss_list.append(loss)
                if loss <= min(loss_list):
                    best_model=copy.deepcopy(model)
                    best_epoch=epoch
                if epoch % self.embed_interval == 0:
                    z,samples,outcomes=project_vae(best_model, train_data if not self.validation_set else self.validation_set, self.cuda)
                    beta_df=pd.DataFrame(z,index=samples)
                    plotly_plot(umap_embed(beta_df,outcomes,n_neighbors=8,supervised=False,min_dist=0.2,metric='euclidean'),'training_{}.html'.format(best_epoch))
            if 0 and self.plot_interval and epoch % self.plot_interval == 0:
                z,_,outcomes=project_vae(model, train_data, self.cuda)
                animation_plts.append(Plot('Latent Embedding, epoch {}'.format(epoch),
                        data=PlotTransformer(z,LabelEncoder().fit_transform(outcomes)).transform()))

        plts=Plotter([Plot(k,'epoch','lr' if 'loss' not in k else k,
                      pd.DataFrame(np.vstack((range(len(plt_data[k])),plt_data[k])).T,
                                   columns=['x','y'])) for k in plt_data],animation=False)
        plts.write_plots(self.loss_plt_fname)
        if 0 and self.plot_interval:
            Plotter(animation_plts).write_plots(self.vae_animation_fname)
        self.model = best_model
        return self

    def add_validation_set(self, validation_data):
        self.validation_set=validation_data

    def transform(self, train_data):
        return project_vae(self.model, train_data, self.cuda)

    def fit_transform(self, train_data):
        return self.fit(train_data).transform(train_data)

def vae_loss(output, input, mean, logvar, loss_func, epoch, kl_warm_up=0, beta=1.):
    recon_loss = loss_func(output, input)
    kl_loss = torch.mean(0.5 * torch.sum(
        torch.exp(logvar) + mean**2 - 1. - logvar, 1))
    kl_loss *= beta
    if epoch < kl_warm_up:
        kl_loss *= np.clip(epoch/kl_warm_up,0.,1.)
    #print(recon_loss,kl_loss)
    return recon_loss + kl_loss, recon_loss, kl_loss

class TybaltTitusVAE(nn.Module):
    def __init__(self, n_input, n_latent, hidden_layer_encoder_topology=[100,100,100], cuda=False):
        super(TybaltTitusVAE, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.cuda_on = cuda
        self.pre_latent_topology = [n_input]+(hidden_layer_encoder_topology if hidden_layer_encoder_topology else [])
        self.post_latent_topology = [n_latent]+(hidden_layer_encoder_topology[::-1] if hidden_layer_encoder_topology else [])
        self.encoder_layers = []
        if len(self.pre_latent_topology)>1:
            for i in range(len(self.pre_latent_topology)-1):
                layer = nn.Linear(self.pre_latent_topology[i],self.pre_latent_topology[i+1])
                torch.nn.init.xavier_uniform(layer.weight)
                self.encoder_layers.append(nn.Sequential(layer,nn.ReLU()))
        self.encoder = nn.Sequential(*self.encoder_layers) if self.encoder_layers else nn.Dropout(p=0.)
        self.z_mean = nn.Sequential(nn.Linear(self.pre_latent_topology[-1],n_latent),nn.BatchNorm1d(n_latent),nn.ReLU())
        self.z_var = nn.Sequential(nn.Linear(self.pre_latent_topology[-1],n_latent),nn.BatchNorm1d(n_latent),nn.ReLU())
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
        if self.cuda_on:
            noise=noise.cuda()
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
    def __init__(self, in_shape, n_latent, cuda=False):
        super(CVAE,self).__init__()
        self.in_shape = in_shape
        self.cuda=cuda
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
        if self.cuda:
            noise=noise.cuda()
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
            inputs = inputs.cuda()
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
