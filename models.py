from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from schedulers import *
from plotter import *
from sklearn.preprocessing import LabelEncoder
from pymethylprocess.visualizations import umap_embed, plotly_plot
import copy

def train_vae(model, loader, loss_func, optimizer, cuda=True, epoch=0, kl_warm_up=0, beta=1.):
    model.train(True) #FIXME
    #print(model)
    total_loss,total_recon_loss,total_kl_loss=0.,0.,0.
    stop_iter = loader.dataset.length // loader.batch_size
    total_loss,total_recon_loss,total_kl_loss=0.,0.,0.
    for i,(inputs, _, _) in enumerate(loader):
        if i == stop_iter:
            break
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
    stop_iter = loader.dataset.length // loader.batch_size
    total_loss,total_recon_loss,total_kl_loss=0.,0.,0.
    for i,(inputs, _, _) in enumerate(loader):
        if i == stop_iter:
            break
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
    print(model)
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
    def __init__(self, autoencoder_model, n_epochs, loss_fn, optimizer, cuda=True, kl_warm_up=0, beta=1.,scheduler_opts={}):
        self.model=autoencoder_model
        #print(self.model)
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
        model = self.model
        best_model=copy.deepcopy(self.model)
        animation_plts=[]
        plt_data={'kl_loss':[],'recon_loss':[],'lr':[],'val_kl_loss':[],'val_recon_loss':[], 'val_loss':[]}
        for epoch in range(self.n_epochs):
            model, loss, recon_loss, kl_loss = train_vae(model, train_data, self.loss_fn, self.optimizer, self.cuda, epoch, self.kl_warm_up, self.beta)
            self.scheduler.step()
            plt_data['kl_loss'].append(kl_loss)
            plt_data['recon_loss'].append(recon_loss)
            plt_data['lr'].append(self.scheduler.get_lr())
            print("Epoch {}: Loss {}, Recon Loss {}, KL-Loss {}".format(epoch,loss,recon_loss,kl_loss))
            if self.validation_set:
                model, val_loss, val_recon_loss, val_kl_loss = val_vae(model, self.validation_set, self.loss_fn, self.optimizer, self.cuda, epoch, self.kl_warm_up, self.beta)
                plt_data['val_kl_loss'].append(val_kl_loss)
                plt_data['val_recon_loss'].append(val_recon_loss)
                plt_data['val_loss'].append(val_loss)
                print("Epoch {}: Val-Loss {}, Val-Recon Loss {}, Val-KL-Loss {}".format(epoch,val_loss,val_recon_loss,val_kl_loss))
            if epoch >= self.kl_warm_up:
                loss = loss if not self.validation_set else val_loss
                loss_list.append(loss)
                if loss <= min(loss_list): # next get models for lowest reconstruction and kl, 3 models
                    best_model=copy.deepcopy(model)#.state_dict())
                    best_epoch=epoch
                if 0 and epoch % self.embed_interval == 0:
                    z,samples,outcomes=project_vae(best_model, train_data if not self.validation_set else self.validation_set, self.cuda)
                    beta_df=pd.DataFrame(z,index=samples)
                    plotly_plot(umap_embed(beta_df,outcomes,n_neighbors=8,supervised=False,min_dist=0.2,metric='euclidean'),'training_{}.html'.format(best_epoch))
            if 0 and self.plot_interval and epoch % self.plot_interval == 0:
                z,_,outcomes=project_vae(model, train_data, self.cuda)
                animation_plts.append(Plot('Latent Embedding, epoch {}'.format(epoch),
                        data=PlotTransformer(z,LabelEncoder().fit_transform(outcomes)).transform()))
        if 0:
            plts=Plotter([Plot(k,'epoch','lr' if 'loss' not in k else k,
                          pd.DataFrame(np.vstack((range(len(plt_data[k])),plt_data[k])).T,
                                       columns=['x','y'])) for k in plt_data if plt_data[k]],animation=False)
            plts.write_plots(self.loss_plt_fname)
        if 0 and self.plot_interval:
            Plotter(animation_plts).write_plots(self.vae_animation_fname)
        self.min_loss = min(np.array(plt_data['kl_loss'])+np.array(plt_data['recon_loss']))
        if self.validation_set:
            self.min_val_loss = plt_data['val_loss'][best_epoch]
            self.min_val_kl_loss = plt_data['val_kl_loss'][best_epoch]
            self.min_val_recon_loss = plt_data['val_recon_loss'][best_epoch]
        else:
            self.min_val_loss, self.min_val_kl_loss, self.min_val_recon_loss  = -1.,-1.,-1.
        self.best_epoch = best_epoch
        self.model = best_model#self.model.load_state_dict(best_model)
        self.training_plot_data = plt_data
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
                torch.nn.init.xavier_uniform_(layer.weight)
                self.encoder_layers.append(nn.Sequential(layer,nn.ReLU()))
        self.encoder = nn.Sequential(*self.encoder_layers) if self.encoder_layers else nn.Dropout(p=0.)
        self.z_mean = nn.Sequential(nn.Linear(self.pre_latent_topology[-1],n_latent),nn.BatchNorm1d(n_latent))
        self.z_var = nn.Sequential(nn.Linear(self.pre_latent_topology[-1],n_latent),nn.BatchNorm1d(n_latent))
        self.z_develop = nn.Linear(n_latent,self.pre_latent_topology[-1])
        self.decoder_layers = []
        if len(self.post_latent_topology)>1:
            for i in range(len(self.post_latent_topology)-1):
                layer = nn.Linear(self.post_latent_topology[i],self.post_latent_topology[i+1])
                torch.nn.init.xavier_uniform_(layer.weight)
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

    def forward_predict(self, x):
        return self.get_latent_z(x)

class ChromosomeVAE(nn.Module):
    """Split up CpGs by chromosome, then run autoencoder on each chromosome, concatenating the latent spaces, parameterizing and running KL loss / reconstruction on that."""
    def __init__(self):
        super(nn.Module,self).__init__()
        print('Not implemented.')
        pass

class CVAE(nn.Module):
    """Add option to feed in Hi-C / ATAC-Seq data."""
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

    def forward_predict(self, x):
        return self.get_latent_z(x)

def train_mlp(model, loader, loss_func, optimizer_vae, optimizer_mlp, cuda=True, categorical=False, train_decoder=False):
    model.train(True)

    #model.vae.eval() also freeze for depth of tuning?
    #print(loss_func)
    stop_iter = loader.dataset.length // loader.batch_size
    running_loss=0.
    running_decoder_loss=0.
    for i,(inputs, samples, y_true) in enumerate(loader): # change dataloder for classification/regression tasks
        #print(samples)
        if inputs.size()[0] == 1 and i == stop_iter:
            break
        inputs = Variable(inputs).view(inputs.size()[0],inputs.size()[1])
        y_true = Variable(y_true)
        if categorical:
            y_true=y_true.argmax(1).long()
        #print(inputs.size())
        if cuda:
            inputs = inputs.cuda()
            y_true = y_true.cuda()
        y_predict, z = model(inputs)
        loss = loss_func(y_predict,y_true)

        optimizer_vae.zero_grad()
        optimizer_mlp.zero_grad()
        loss.backward()
        if train_decoder:
            model, decoder_loss = train_decoder_(model, inputs, z)
            running_decoder_loss += decoder_loss
        optimizer_vae.step()
        optimizer_mlp.step()
        running_loss+=loss.item()
    if train_decoder:
        print('Decoder Loss is {}'.format(running_decoder_loss))
    return model, running_loss

def val_mlp(model, loader, loss_func, cuda=True, categorical=False, train_decoder=False):
    model.eval()

    #model.vae.eval() also freeze for depth of tuning?
    stop_iter = loader.dataset.length // loader.batch_size
    running_decoder_loss=0.
    running_loss=0.
    for i,(inputs, samples, y_true) in enumerate(loader): # change dataloder for classification/regression tasks
        if inputs.size()[0] == 1 and i == stop_iter:
            break
        inputs = Variable(inputs).view(inputs.size()[0],inputs.size()[1])
        y_true = Variable(y_true)
        #print(inputs.size())
        if categorical:
            y_true=y_true.argmax(1).long()
        if cuda:
            inputs = inputs.cuda()
            y_true = y_true.cuda()
        y_predict, z = model(inputs)
        loss = loss_func(y_predict,y_true)
        running_loss+=loss.item()
        if train_decoder:
            running_decoder_loss += val_decoder_(model, inputs, z)
    if train_decoder:
        print('Val Decoder Loss is {}'.format(running_decoder_loss))
    return model, running_loss

def test_mlp(model, loader, categorical, cuda=True, output_latent=True):
    model.eval()
    #print(model)
    Y_pred=[]
    final_latent=[]
    sample_names_final=[]
    Y_true=[]
    for inputs, sample_names, y_true in loader: # change dataloder for classification/regression tasks
        print(inputs)
        inputs = Variable(inputs).view(inputs.size()[0],inputs.size()[1])
        y_true = Variable(y_true)
        #print(inputs.size())
        if cuda:
            inputs = inputs.cuda()
            y_true = y_true.cuda()
        y_predict, z = model(inputs)
        y_predict=np.squeeze(y_predict.detach().cpu().numpy())
        y_true=np.squeeze(y_true.detach().cpu().numpy())
        #print(y_predict.shape,y_true.shape)
        #print(y_predict,y_true)
        if len(y_predict.shape) < 2:
            y_predict=y_predict.flatten()
        if len(y_true.shape) < 2:
            y_true=y_true.flatten()  # FIXME
        Y_pred.append(y_predict)
        final_latent.append(np.squeeze(z.detach().cpu().numpy()))
        sample_names_final.extend([name[0] for name in sample_names])
        Y_true.append(y_true)
    if len(Y_pred) > 1:
        if all(list(map(lambda x: len(np.shape(x))<2,Y_pred))):
            Y_pred = np.hstack(Y_pred)[:,np.newaxis]
        else:
            Y_pred=np.vstack(Y_pred)
    else:
        Y_pred = Y_pred[0]
        if len(np.shape(Y_pred))<2:
            Y_pred=Y_pred[:,np.newaxis]
    if len(final_latent) > 1:

        final_latent=np.vstack(final_latent)
    else:
        final_latent = final_latent[0]
    if len(Y_true) > 1:
        if all(list(map(lambda x: len(np.shape(x))<2,Y_true))):
            Y_true = np.hstack(Y_true)[:,np.newaxis]
        else:
            Y_true=np.vstack(Y_true)
    else:
        Y_true = Y_true[0]
        if len(np.shape(Y_true))<2:
            Y_true=Y_true[:,np.newaxis]
    print(Y_pred,Y_true)
    #print(np.hstack([Y_pred,Y_true]))
    sample_names_final = np.array(sample_names_final)
    if output_latent:
        return Y_pred, Y_true, final_latent, sample_names_final
    else:
        return Y_pred

def train_decoder_(model, x, z):
    model.vae.train(True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.vae.decoder.parameters():
        param.requires_grad = True
    loss_fn = nn.BCELoss(reduction='sum')
    x_hat = model.decode(z)
    if type(x_hat) != type([]):
        x_hat = [x_hat]
    loss = sum([loss_func(x_h, x) for x_h in x_hat])
    loss.backward()
    for param in model.parameters():
        param.requires_grad = True
    model.vae.eval()
    return model, loss.item()

def val_decoder_(model, x, z):
    model.vae.eval()
    loss_fn = nn.BCELoss(reduction='sum')
    x_hat = model.decode(z)
    if type(x_hat) != type([]):
        x_hat = [x_hat]
    loss = sum([loss_func(x_h, x) for x_h in x_hat])
    return loss.item()


class MLPFinetuneVAE:
    def __init__(self, mlp_model, n_epochs=None, loss_fn=None, optimizer_vae=None, optimizer_mlp=None, cuda=True, categorical=False, scheduler_opts={}, output_latent=True, train_decoder=False):
        self.model=mlp_model
        #print(self.model)
        if cuda:
            self.model = self.model.cuda()
            #self.model.vae = self.model.vae.cuda()
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.optimizer_vae = optimizer_vae
        self.optimizer_mlp = optimizer_mlp
        self.cuda = cuda
        if self.optimizer_vae!=None and self.optimizer_mlp!=None:
            self.scheduler_vae = Scheduler(self.optimizer_vae,scheduler_opts) if scheduler_opts else Scheduler(self.optimizer_vae)
            self.scheduler_mlp = Scheduler(self.optimizer_mlp,scheduler_opts) if scheduler_opts else Scheduler(self.optimizer_mlp)
        else:
            self.scheduler_vae = None
            self.scheduler_vae = None
        self.loss_plt_fname='loss.png'
        self.embed_interval=200
        self.validation_set = False
        self.return_latent = True
        self.categorical = categorical
        self.output_latent = output_latent
        self.train_decoder = train_decoder # FIXME add loss for decoder if selecting this option and freeze other weights when updating decoder, also change forward function to include reconstruction, change back when done

    def fit(self, train_data):
        loss_list = []
        model = self.model
        print(model)
        best_model=copy.deepcopy(self.model)
        plt_data={'loss':[],'lr_vae':[],'lr_mlp':[], 'val_loss':[]}
        for epoch in range(self.n_epochs):
            print(epoch)
            model, loss = train_mlp(model, train_data, self.loss_fn, self.optimizer_vae, self.optimizer_mlp, self.cuda,categorical=self.categorical, train_decoder=self.train_decoder)
            self.scheduler_vae.step()
            self.scheduler_mlp.step()
            plt_data['loss'].append(loss)
            print("Epoch {}: Loss {}".format(epoch,loss))
            if self.validation_set:
                model, val_loss = val_mlp(model, self.validation_set, self.loss_fn, self.cuda,categorical=self.categorical, train_decoder=self.train_decoder)
                plt_data['val_loss'].append(val_loss)
                print("Epoch {}: Val-Loss {}".format(epoch,val_loss))
            plt_data['lr_vae'].append(self.scheduler_vae.get_lr())
            plt_data['lr_mlp'].append(self.scheduler_mlp.get_lr())
            loss = loss if not self.validation_set else val_loss
            loss_list.append(loss)
            if loss <= min(loss_list): # next get models for lowest reconstruction and kl, 3 models
                best_model=copy.deepcopy(model)
                best_epoch=epoch
        self.training_plot_data=plt_data
        if 0:
            plts=Plotter([Plot(k,'epoch','lr' if 'loss' not in k else k,
                          pd.DataFrame(np.vstack((range(len(plt_data[k])),plt_data[k])).T,
                                       columns=['x','y'])) for k in plt_data if plt_data[k]],animation=False)
            plts.write_plots(self.loss_plt_fname)
        self.min_loss = min(plt_data['loss'])
        if self.validation_set:
            self.min_val_loss = min(plt_data['val_loss'])
        else:
            self.min_val_loss = -1
        self.best_epoch = best_epoch
        self.model = best_model
        return self

    def add_validation_set(self, validation_data):
        self.validation_set=validation_data

    def predict(self, test_data):
        return test_mlp(self.model, test_data, self.categorical, self.cuda, self.output_latent)

class VAE_MLP(nn.Module): # add ability to train decoderF
    def __init__(self, vae_model, n_output, categorical=False, hidden_layer_topology=[100,100,100], dropout_p=0.2, add_softmax=False):
        super(VAE_MLP,self).__init__()
        self.vae = vae_model
        self.n_output = n_output
        self.categorical = categorical
        self.add_softmax = add_softmax
        self.topology = [self.vae.n_latent]+(hidden_layer_topology if hidden_layer_topology else [])
        self.mlp_layers = []
        self.dropout_p=dropout_p
        if len(self.topology)>1:
            for i in range(len(self.topology)-1):
                layer = nn.Linear(self.topology[i],self.topology[i+1])
                torch.nn.init.xavier_uniform_(layer.weight)
                self.mlp_layers.append(nn.Sequential(layer,nn.ReLU(),nn.Dropout(self.dropout_p)))
        self.output_layer = nn.Linear(self.topology[-1],self.n_output)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.mlp_layers.extend([self.output_layer]+([nn.Softmax()] if self.add_softmax else []))#+([nn.LogSoftmax()] if self.categorical else []))
        self.mlp = nn.Sequential(*self.mlp_layers)
        self.output_z=False

    def forward(self,x):
        z=self.vae.get_latent_z(x)
        return self.mlp(z), z

    def decode(self,z):
        return self.vae.decoder(z)

    def forward_embed(self,x):
        out=self.vae.get_latent_z(x)
        recon=self.vae.decoder(out)
        return self.mlp(out), out, recon

    def toggle_latent_z(self):
        if self.output_z:
            self.output_z=False
        else:
            self.output_z=True

    def forward_predict(self,x):
        if self.output_z:
            return self.vae.get_latent_z(x)
        else:
            return self.mlp(self.vae.get_latent_z(x))
