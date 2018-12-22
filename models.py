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

def val_vae(model, loader, loss_func, optimizer, cuda=True, epoch=0, kl_warm_up=0, beta=1.):
    model.eval() #FIXME
    #print(model)
    total_loss,total_recon_loss,total_kl_loss=0.,0.,0.
    for inputs, _, _ in loader:
        inputs = Variable(inputs).view(inputs.size()[0],inputs.size()[1]) # modify for convolutions, add batchnorm2d?
        #print(inputs.size())
        if cuda:
            inputs = inputs.cuda()
        output, mean, logvar = model(inputs)
        loss, reconstruction_loss, kl_loss = vae_loss(output, inputs, mean, logvar, loss_func, epoch, kl_warm_up, beta)
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
        plt_data={'kl_loss':[],'recon_loss':[],'lr':[],'val_kl_loss':[],'val_recon_loss':[], 'val_loss':[]}
        for epoch in range(self.n_epochs):
            model, loss, recon_loss, kl_loss = train_vae(self.model, train_data, self.loss_fn, self.optimizer, self.cuda, epoch, self.kl_warm_up, self.beta)
            self.scheduler.step()
            plt_data['kl_loss'].append(kl_loss)
            plt_data['recon_loss'].append(recon_loss)
            plt_data['lr'].append(self.scheduler.get_lr())
            if self.validation_set:
                _, val_loss, val_recon_loss, val_kl_loss = val_vae(self.model, self.validation_set, self.loss_fn, self.optimizer, self.cuda, epoch, self.kl_warm_up, self.beta)
                plt_data['val_kl_loss'].append(val_kl_loss)
                plt_data['val_recon_loss'].append(val_recon_loss)
                plt_data['val_loss'].append(val_loss)
            print("Epoch {}: Loss {}, Recon Loss {}, KL-Loss {}".format(epoch,loss,recon_loss,kl_loss))
            if epoch > self.kl_warm_up:
                loss_list.append(loss)
                if loss <= min(loss_list): # next get models for lowest reconstruction and kl, 3 models
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
                                   columns=['x','y'])) for k in plt_data if plt_data[k]],animation=False)
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
    if type(output) != type([]):
        output = [output]
    recon_loss = sum([loss_func(out, input) for out in output])
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
        self.z_mean = nn.Sequential(nn.Linear(self.pre_latent_topology[-1],n_latent),nn.BatchNorm1d(n_latent))
        self.z_var = nn.Sequential(nn.Linear(self.pre_latent_topology[-1],n_latent),nn.BatchNorm1d(n_latent))
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
    def __init__(self, in_shape, n_latent, custom_kernel_sizes, kernel_widths, kernel_heights, n_pre_latent, stride_size, cuda=False):
        super(CVAE,self).__init__()
        self.n_post_latent = n_pre_latent
        self.in_shape = in_shape
        self.cuda=cuda
        self.n_latent = n_latent
        c,h,w = in_shape
        self.w_kernel_sizes = [(h,width) for width in kernel_widths]
        self.h_kernel_sizes = [(w,height) for height in kernel_heights]
        self.custom_kernel_sizes = custom_kernel_sizes
        self.z_dim = h//2**2 # receptive field downsampled 2 times
        encoder_generate = lambda kernel_size: nn.Sequential(
            nn.BatchNorm2d(c),
            nn.Conv2d(c, 32, kernel_size=kernel_size, stride=stride_size, padding=1),  # 32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride_size, padding=1),  # 32, 8, 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Linear(64 * self.z_dim**2, n_pre_latent),
            nn.ReLU()
        )
        self.n_feature_maps = len(self.w_kernel_sizes)+len(self.h_kernel_sizes)+len(self.custom_kernel_sizes)
        self.z_mean = nn.Sequential(nn.Linear(n_pre_latent*self.n_feature_maps,n_latent),nn.BatchNorm1d(n_latent))
        self.z_var = nn.Sequential(nn.Linear(n_pre_latent*self.n_feature_maps,n_latent),nn.BatchNorm1d(n_latent))
        self.z_develop = nn.Linear(n_latent, self.n_post_latent*self.n_feature_maps)
        decoder_generate = lambda kernel_size: nn.Sequential(
            nn.Linear(n_post_latent, 64 * self.z_dim**2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride_size, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, stride=stride_size, padding=1),
            CenterCrop(h,w),
            nn.Sigmoid()
        )
        self.encoders = [encoder_generate(kernel_size) for kernel_size in self.custom_kernel_sizes+self.w_kernel_sizes+self.h_kernel_sizes]
        self.decoders = [decoder_generate(kernel_size) for kernel_size in self.custom_kernel_sizes+self.w_kernel_sizes+self.h_kernel_sizes]

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = Variable(torch.randn(stddev.size()))
        if self.cuda:
            noise=noise.cuda()
        return (noise * stddev) + mean

    def encode(self, x):
        encoder_outputs = torch.cat([encoder(x) for encoder in self.encoders],axis=1)
        x = x.view(x.size(0), -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    def decode(self, x):
        out = self.z_develop(z)
        outs = torch.chunk(z,self.n_feature_maps,dim=1)
        outs = [decoder(out.view(out.size(0), 64, self.z_dim, self.z_dim)) for out in outs]
        return outs

    def encode_old(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    def decode_old(self, z):
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

def train_mlp(model, loader, loss_func, optimizer, cuda=True):
    model.train(True)

    #model.vae.eval() also freeze for depth of tuning?

    for inputs, samples, y_true in loader: # change dataloder for classification/regression tasks
        inputs = Variable(inputs).view(inputs.size()[1],-1,inputs.size()[2])
        #print(inputs.size())
        if cuda:
            inputs = inputs.cuda()
        y_predict, _ = model(inputs)
        loss = loss_func(y_predict,y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, loss

def val_mlp(model, loader, loss_func, optimizer, cuda=True):
    model.eval()

    #model.vae.eval() also freeze for depth of tuning?

    for inputs, samples, y_true in loader: # change dataloder for classification/regression tasks
        inputs = Variable(inputs).view(inputs.size()[1],-1,inputs.size()[2])
        #print(inputs.size())
        if cuda:
            inputs = inputs.cuda()
        y_predict, _ = model(inputs)
        loss = loss_func(y_predict,y_true)

    return model, loss

def test_mlp(model, loader, categorical, cuda=True):
    model.eval()
    #print(model)
    Y_pred=[]
    final_latent=[]
    sample_names_final=[]
    Y_true=[]
    for inputs, sample_names, y_true in loader: # change dataloder for classification/regression tasks
        inputs = Variable(inputs).view(inputs.size()[1],-1,inputs.size()[2])
        #print(inputs.size())
        if cuda:
            inputs = inputs.cuda()
        y_predict, z = model(inputs)
        y_predict=np.squeeze(y_predict.detach().cpu().numpy())
        y_true=np.squeeze(y_true.detach().cpu().numpy())
        if categorical:
            y_predict=y_predict.argmax(axis=1)[:,np.newaxis]
            y_true=y_true.argmax(axis=1)[:,np.newaxis]
        Y_pred.append(y_predict)
        final_latent.append(np.squeeze(z.detach().cpu().numpy()))
        sample_names_final.extend([name[0] for name in sample_names])
        Y_true.append(y_true)
    Y_pred=np.vstack(Y_pred)
    final_latent=np.vstack(final_latent)
    Y_true=np.vstack(final_outputs)
    sample_names_final = np.array(sample_names_final)
    return Y_pred, Y_true, final_latent, sample_names_final

class MLPFinetuneVAE:
    def __init__(self, mlp_model, n_epochs, loss_fn, optimizer, cuda=True, categorical=False, scheduler_opts={}):
        self.model=mlp_model
        print(self.model)
        if cuda:
            self.model = self.model.cuda()
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cuda = cuda
        self.scheduler = Scheduler(self.optimizer,scheduler_opts) if scheduler_opts else Scheduler(self.optimizer)
        self.loss_plt_fname='loss.png'
        self.embed_interval=200
        self.validation_set = False
        self.return_latent = True
        self.categorical = categorical

    def fit(self, train_data):
        loss_list = []
        best_model=None
        plt_data={'loss':[],'lr':[], 'val_loss':[]}
        for epoch in range(self.n_epochs):
            model, loss = train_mlp(self.model, train_data, self.loss_fn, self.optimizer, self.cuda)
            self.scheduler.step()
            plt_data['loss'].append(loss)
            if self.validation_set:
                _, val_loss = val_mlp(self.model, self.validation_set, self.loss_fn, self.optimizer, self.cuda)
                plt_data['val_loss'].append(val_loss)
            plt_data['lr'].append(self.scheduler.get_lr())
            print("Epoch {}: Loss {}".format(epoch,loss))
            loss_list.append(loss)
            if loss <= min(loss_list): # next get models for lowest reconstruction and kl, 3 models
                best_model=copy.deepcopy(model)
                best_epoch=epoch

        plts=Plotter([Plot(k,'epoch','lr' if 'loss' not in k else k,
                      pd.DataFrame(np.vstack((range(len(plt_data[k])),plt_data[k])).T,
                                   columns=['x','y'])) for k in plt_data if plt_data[k]],animation=False)
        plts.write_plots(self.loss_plt_fname)
        self.model = best_model
        return self

    def add_validation_set(self, validation_data):
        self.validation_set=validation_data

    def predict(self, test_data):
        return test_mlp(self.model, test_data, self.categorical, self.cuda)

class VAE_MLP(nn.Module):
    def __init__(self, vae_model, n_output, categorical=False, hidden_layer_topology=[100,100,100]):
        super(VAE_MLP,self).__init__()
        self.vae = vae_model.vae_model
        self.n_output = n_output
        self.categorical = categorical
        self.topology = [self.n_latent]+(hidden_layer_topology if hidden_layer_topology else [])
        self.mlp_layers = []
        if len(self.topology)>1:
            for i in range(len(self.topology)-1):
                layer = nn.Linear(self.topology[i],self.topology[i+1])
                torch.nn.init.xavier_uniform(layer.weight)
                self.mlp_layers.append(nn.Sequential(layer,nn.ReLU()))
        self.output_layer = nn.Linear(self.topology[-1],self.n_output)
        torch.nn.init.xavier_uniform(self.output_layer.weight)
        self.mlp_layers.extend([self.output_layer]+([nn.Sigmoid()] if self.categorical else [] ))
        self.mlp = self.Sequential(*self.mlp_layers)

    def forward(self,x):
        out=self.vae.get_latent_z(x)
        return self.mlp(out), out
