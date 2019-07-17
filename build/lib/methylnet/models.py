"""
models.py
=======================
Contains core PyTorch Models for running VAE and VAE-MLP."""

from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from methylnet.schedulers import *
from methylnet.plotter import *
from sklearn.preprocessing import LabelEncoder
from pymethylprocess.visualizations import umap_embed, plotly_plot
import copy

def train_vae(model, loader, loss_func, optimizer, cuda=True, epoch=0, kl_warm_up=0, beta=1.):
    """Function for parameter update during VAE training for one iteration.

    Parameters
    ----------
    model : type
        VAE torch model
    loader : type
        Data loader, generator that calls batches of data.
    loss_func : type
        Loss function for reconstruction error, nn.BCELoss or MSELoss
    optimizer : type
        SGD or Adam pytorch optimizer.
    cuda : type
        GPU?
    epoch : type
        Epoch of training, passed in from outer loop.
    kl_warm_up : type
        How many epochs until model is fully utilizes KL Loss.
    beta : type
        Weight given to KL Loss.

    Returns
    -------
    nn.Module
        Pytorch VAE model
    float
        Total Training Loss across all batches
    float
        Total Training reconstruction loss across all batches
    float
        Total KL Loss across all batches
    """
    model.train(True) #FIXME
    #print(model)
    total_loss,total_recon_loss,total_kl_loss=0.,0.,0.
    stop_iter = loader.dataset.length // loader.batch_size
    total_loss,total_recon_loss,total_kl_loss=0.,0.,0.
    for i,(inputs, _) in enumerate(loader):
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
    """Function for validation loss computation during VAE training for one epoch.

    Parameters
    ----------
    model : type
        VAE torch model
    loader : type
        Validation Data loader, generator that calls batches of data.
    loss_func : type
        Loss function for reconstruction error, nn.BCELoss or MSELoss
    optimizer : type
        SGD or Adam pytorch optimizer.
    cuda : type
        GPU?
    epoch : type
        Epoch of training, passed in from outer loop.
    kl_warm_up : type
        How many epochs until model is fully utilizes KL Loss.
    beta : type
        Weight given to KL Loss.

    Returns
    -------
    nn.Module
        Pytorch VAE model
    float
        Total Validation Loss across all batches
    float
        Total Validation reconstruction loss across all batches
    float
        Total Validation KL Loss across all batches
    """
    model.eval() #FIXME
    #print(model)
    stop_iter = loader.dataset.length // loader.batch_size
    total_loss,total_recon_loss,total_kl_loss=0.,0.,0.
    with torch.no_grad():
        for i,(inputs, _) in enumerate(loader):
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
    """Return Latent Embeddings of any data supplied to it.

    Parameters
    ----------
    model : type
        VAE Pytorch Model.
    loader : type
        Loads data one batch at a time.
    cuda : type
        GPU?

    Returns
    -------
    np.array
        Latent Embeddings.
    np.array
        Sample names from MethylationArray
    np.array
        Outcomes from column of methylarray.
    """
    print(model)
    model.eval()
    #print(model)
    final_outputs=[]
    #outcomes_final=[]
    with torch.no_grad():
        for inputs, outcomes in loader:
            inputs = Variable(inputs).view(inputs.size()[0],inputs.size()[1]) # modify for convolutions, add batchnorm2d?
            if cuda:
                inputs = inputs.cuda()
            z = np.squeeze(model.get_latent_z(inputs).detach().cpu().numpy())
            final_outputs.append(z)
            #outcomes_final.extend([outcome[0] for outcome in outcomes])
        z=np.vstack(final_outputs)
        #sample_names=np.array(sample_names_final)
        #outcomes=np.array(outcomes_final)
    return z, None, None

class AutoEncoder:
    """Wraps Pytorch VAE module into Scikit-learn like interface for ease of training, validation and testing.

    Parameters
    ----------
    autoencoder_model : type
        Pytorch VAE Model to supply.
    n_epochs : type
        Number of epochs to train for.
    loss_fn : type
        Pytorch loss function for reconstruction error.
    optimizer : type
        Pytorch Optimizer.
    cuda : type
        GPU?
    kl_warm_up : type
        Number of epochs until fully utilizing KLLoss, begin saving models here.
    beta : type
        Weighting for KLLoss.
    scheduler_opts : type
        Options to feed learning rate scheduler, which modulates learning rate of optimizer.

    Attributes
    ----------
    model : type
        Pytorch VAE model.
    scheduler : type
        Learning rate scheduler object.
    vae_animation_fname : type
        Save VAE embeddings evolving over epochs to this file name. Defunct for now.
    loss_plt_fname : type
        Where to save loss curves. This has been superceded by plot_training_curves in methylnet-visualize command.
    plot_interval : type
        How often to plot data; defunct.
    embed_interval : type
        How often to embed; defunct.
    validation_set : type
        MethylationArray DataLoader, produced from Pytorch MethylationDataset of Validation MethylationArray.
    n_epochs
    loss_fn
    optimizer
    cuda
    kl_warm_up
    beta

    """
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
        """Fit VAE model to training data, best model returned with lowest validation loss over epochs.

        Parameters
        ----------
        train_data : DataLoader
            Training DataLoader that is loading MethylationDataset in batches.

        Returns
        -------
        self
            Autoencoder object with updated VAE model.

        """
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
        """Add validation data in the form of Validation DataLoader. Adding this will use validation data for early termination / generalization of model to unseen data.

        Parameters
        ----------
        validation_data : type
            Pytorch DataLoader housing validation MethylationDataset.

        """
        self.validation_set=validation_data

    def transform(self, train_data):
        """

        Parameters
        ----------
        train_data : type
            Pytorch DataLoader housing training MethylationDataset.

        Returns
        -------
        np.array
            Latent Embeddings.
        np.array
            Sample names from MethylationArray
        np.array
            Outcomes from column of methylarray.


        """
        return project_vae(self.model, train_data, self.cuda)

    def fit_transform(self, train_data):
        """Fit VAE model and transform Methylation Array using VAE model.

        Parameters
        ----------
        train_data : type
            Pytorch DataLoader housing training MethylationDataset.

        Returns
        -------
        np.array
            Latent Embeddings.
        np.array
            Sample names from MethylationArray
        np.array
            Outcomes from column of methylarray.

        """
        return self.fit(train_data).transform(train_data)

def vae_loss(output, input, mean, logvar, loss_func, epoch, kl_warm_up=0, beta=1.):
    """Function to calculate VAE Loss, Reconstruction Loss + Beta KLLoss.

    Parameters
    ----------
    output : torch.tensor
        Reconstructed output from autoencoder.
    input : torch.tensor
        Original input data.
    mean : type
        Learned mean tensor for each sample point.
    logvar : type
        Variation around that mean sample point, learned from reparameterization.
    loss_func : type
        Loss function for reconstruction loss, MSE or BCE.
    epoch : type
        Epoch of training.
    kl_warm_up : type
        Number of epochs until fully utilizing KLLoss, begin saving models here.
    beta : type
        Weighting for KLLoss.

    Returns
    -------
    torch.tensor
        Total loss
    torch.tensor
        Recon loss
    torch.tensor
        KL loss

    """
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
    """Pytorch NN Module housing VAE with fully connected layers and customizable topology.

    Parameters
    ----------
    n_input : type
        Number of input CpGs.
    n_latent : type
        Size of latent embeddings.
    hidden_layer_encoder_topology : type
        List, length of list contains number of hidden layers for encoder, and each element is number of neurons, mirrored for decoder.
    cuda : type
        GPU?

    Attributes
    ----------
    cuda_on : type
        GPU?
    pre_latent_topology : type
        Hidden layer topology for encoder.
    post_latent_topology : type
        Mirrored hidden layer topology for decoder.
    encoder_layers : list
        Encoder pytorch layers.
    encoder : type
        Encoder layers wrapped into pytorch module.
    z_mean : type
        Linear layer from last encoder layer to mean layer.
    z_var : type
        Linear layer from last encoder layer to var layer.
    z_develop : type
        Linear layer connecting sampled latent embedding to first layer decoder.
    decoder_layers : type
        Decoder layers wrapped into pytorch module.
    output_layer : type
        Linear layer connecting last decoder layer to output layer, which is same size as input..
    decoder : type
        Wraps decoder_layers and output_layers into Sequential module.
    n_input
    n_latent

    """
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
        """Sample latent embeddings, reparameterize by adding noise to embedding.

        Parameters
        ----------
        mean : type
            Learned mean vector of embeddings.
        logvar : type
            Learned variance of learned mean embeddings.

        Returns
        -------
        torch.tensor
            Mean + noise, reparameterization trick.

        """
        stddev = torch.exp(0.5 * logvar)
        noise = Variable(torch.randn(stddev.size()))
        if self.cuda_on:
            noise=noise.cuda()
        if not self.training:
            noise = 0.
            stddev = 0.
        return (noise * stddev) + mean

    def encode(self, x):
        """Encode input into latent representation.

        Parameters
        ----------
        x : type
            Input methylation data.

        Returns
        -------
        torch.tensor
            Learned mean vector of embeddings.
        torch.tensor
            Learned variance of learned mean embeddings.
        """
        x = self.encoder(x)
        #print(x.size())
        #x = x.view(x.size(0), -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        #print('mean',mean.size())
        return mean, var

    def decode(self, z):
        """Decode latent embeddings back into reconstructed input.

        Parameters
        ----------
        z : type
            Reparameterized latent embedding.

        Returns
        -------
        torch.tensor
            Reconstructed input.

        """
        #out = self.z_develop(z)
        #print('out',out.size())
        #out = out.view(z.size(0), 64, self.z_dim, self.z_dim)
        out = self.decoder(z)
        #print(out)
        return out

    def forward(self, x):
        """Return reconstructed output, mean and variance of embeddings.
        """
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar

    def get_latent_z(self, x):
        """Encode X into reparameterized latent representation.

        Parameters
        ----------
        x : type
            Input methylation data.

        Returns
        -------
        torch.tensor
            Latent embeddings.

        """
        mean, logvar = self.encode(x)
        return self.sample_z(mean, logvar)

    def forward_predict(self, x):
        """Forward pass from input to reconstructed input."""
        return self.get_latent_z(x)

def train_mlp(model, loader, loss_func, optimizer_vae, optimizer_mlp, cuda=True, categorical=False, train_decoder=False):
    """Train Multi-layer perceptron appended to latent embeddings of VAE via transfer learning. Do this for one iteration.

    Parameters
    ----------
    model : type
        VAE_MLP model.
    loader : type
        DataLoader with MethylationDataset.
    loss_func : type
        Loss function (BCE, CrossEntropy, MSE).
    optimizer_vae : type
        Optimizer for pytorch VAE.
    optimizer_mlp : type
        Optimizer for outcome MLP layers.
    cuda : type
        GPU?
    categorical : type
        Predicting categorical or continuous outcomes.
    train_decoder : type
        Retrain decoder during training loop to adjust for fine-tuned embeddings.

    Returns
    -------
    nn.Module
        Training VAE_MLP model with updated parameters.
    float
        Training loss over all batches

    """
    model.train(True)

    #model.vae.eval() also freeze for depth of tuning?
    #print(loss_func)
    stop_iter = loader.dataset.length // loader.batch_size
    running_loss=0.
    running_decoder_loss=0.
    for i,(inputs, y_true) in enumerate(loader): # change dataloder for classification/regression tasks
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
    """Find validation loss of VAE_MLP over one Epoch.

    Parameters
    ----------
    model : type
        VAE_MLP model.
    loader : type
        DataLoader with MethylationDataset.
    loss_func : type
        Loss function (BCE, CrossEntropy, MSE).
    cuda : type
        GPU?
    categorical : type
        Predicting categorical or continuous outcomes.
    train_decoder : type
        Retrain decoder during training loop to adjust for fine-tuned embeddings.

    Returns
    -------
    nn.Module
        VAE_MLP model.
    float
        Validation loss over all batches

    """
    model.eval()

    #model.vae.eval() also freeze for depth of tuning?
    stop_iter = loader.dataset.length // loader.batch_size
    running_decoder_loss=0.
    running_loss=0.
    with torch.no_grad():
        for i,(inputs, y_true) in enumerate(loader): # change dataloder for classification/regression tasks
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
    """Evaluate MLP on testing set, output predictions.

    Parameters
    ----------
    model : type
        VAE_MLP model.
    loader : type
        DataLoader with MethylationDataSet
    categorical : type
        Categorical or continuous predictions.
    cuda : type
        GPU?
    output_latent : type
        Output latent embeddings in addition to predictions?

    Returns
    -------
    np.array
        Predictions
    np.array
        Ground truth
    np.array
        Latent Embeddings
    np.array
        Sample names.

    """
    model.eval()
    #print(model)
    Y_pred=[]
    final_latent=[]
    Y_true=[]
    with torch.no_grad():
        for inputs, y_true in loader: # change dataloder for classification/regression tasks
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
    if output_latent:
        return Y_pred, Y_true, final_latent, None
    else:
        return Y_pred

def train_decoder_(model, x, z):
    """Run if retraining decoder to adjust for adjusted latent embeddings during finetuning of embedding layers for VAE_MLP.

    Parameters
    ----------
    model : type
        VAE_MLP model.
    x : type
        Input methylation data.
    z : type
        Latent Embeddings

    Returns
    -------
    nn.Module
        VAE_MLP module with updated decoder parameters.
    float
        Reconstruction loss over all batches.

    """
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
    """Validation Loss over decoder.

    Parameters
    ----------
    model : type
        VAE_MLP model.
    x : type
        Input methylation data.
    z : type
        Latent Embeddings

    Returns
    -------
    float
        Reconstruction loss over all batches.
    """
    model.vae.eval()
    loss_fn = nn.BCELoss(reduction='sum')
    x_hat = model.decode(z)
    if type(x_hat) != type([]):
        x_hat = [x_hat]
    loss = sum([loss_func(x_h, x) for x_h in x_hat])
    return loss.item()


class MLPFinetuneVAE:
    """Wraps VAE_MLP pytorch module into scikit-learn interface with fit, predict and fit_predict methods for ease-of-use model training/evaluation.

    Parameters
    ----------
    mlp_model : type
        VAE_MLP model.
    n_epochs : type
        Number epochs train for.
    loss_fn : type
        Loss function, pytorch, CrossEntropy, BCE, MSE depending on outcome.
    optimizer_vae : type
        Optimizer for VAE layers for finetuning original pretrained network.
    optimizer_mlp : type
        Optimizer for new appended MLP layers.
    cuda : type
        GPU?
    categorical : type
        Classification or regression outcome?
    scheduler_opts : type
        Options for learning rate scheduler, modulates learning rates for VAE and MLP.
    output_latent : type
        Whether to output latent embeddings during evaluation.
    train_decoder : type
        Retrain decoder to adjust for finetuning of VAE?

    Attributes
    ----------
    model : type
        VAE_MLP.
    scheduler_vae : type
        Learning rate modulator for VAE optimizer.
    scheduler_mlp : type
        Learning rate modulator for MLP optimizer.
    loss_plt_fname : type
        File where to plot loss over time; defunct.
    embed_interval : type
        How often to return embeddings; defunct.
    validation_set : type
        Validation set used for hyperparameter tuning and early stopping criteria for generalization.
    return_latent : type
        Return embedding during evaluation?
    n_epochs
    loss_fn
    optimizer_vae
    optimizer_mlp
    cuda
    categorical
    output_latent
    train_decoder

    """
    def __init__(self, mlp_model, n_epochs=None, loss_fn=None, optimizer_vae=None, optimizer_mlp=None, cuda=True, categorical=False, scheduler_opts={}, output_latent=True, train_decoder=False):
        self.model=mlp_model
        #print(self.model)
        self.model.vae.cuda_on = cuda
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
            self.scheduler_mlp = None
        self.loss_plt_fname='loss.png'
        self.embed_interval=200
        self.validation_set = False
        self.return_latent = True
        self.categorical = categorical
        self.output_latent = output_latent
        self.train_decoder = train_decoder # FIXME add loss for decoder if selecting this option and freeze other weights when updating decoder, also change forward function to include reconstruction, change back when done
        self.train_fn = train_mlp
        self.val_fn = val_mlp
        self.test_fn = test_mlp

    def fit(self, train_data):
        """Fit MLP to training data to make predictions.

        Parameters
        ----------
        train_data : type
            DataLoader with Training MethylationDataset.

        Returns
        -------
        self
            MLPFinetuneVAE with updated parameters.
        """
        loss_list = []
        model = self.model
        print(model)
        best_model=copy.deepcopy(self.model)
        plt_data={'loss':[],'lr_vae':[],'lr_mlp':[], 'val_loss':[]}
        for epoch in range(self.n_epochs):
            print(epoch)
            model, loss = self.train_fn(model, train_data, self.loss_fn, self.optimizer_vae, self.optimizer_mlp, self.cuda,categorical=self.categorical, train_decoder=self.train_decoder)
            self.scheduler_vae.step()
            self.scheduler_mlp.step()
            plt_data['loss'].append(loss)
            print("Epoch {}: Loss {}".format(epoch,loss))
            if self.validation_set:
                model, val_loss = self.val_fn(model, self.validation_set, self.loss_fn, self.cuda,categorical=self.categorical, train_decoder=self.train_decoder)
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
        """Add validation data to reduce overfitting.

        Parameters
        ----------
        validation_data : type
            Validation Dataloader MethylationDataset.

        """
        self.validation_set=validation_data

    def predict(self, test_data):
        """Short summary.

        Parameters
        ----------
        test_data : type
            Test DataLoader MethylationDataset.

        Returns
        -------
        np.array
            Predictions
        np.array
            Ground truth
        np.array
            Latent Embeddings
        np.array
            Sample names.

        """
        return self.test_fn(self.model, test_data, self.categorical, self.cuda, self.output_latent)

class VAE_MLP(nn.Module):
    """VAE_MLP, pytorch module used to both finetune VAE embeddings and simultaneously train downstream MLP layers for classification/regression tasks.

    Parameters
    ----------
    vae_model : type
        VAE pytorch model for methylation data.
    n_output : type
        Number of outputs at end of model.
    categorical : type
        Classification or regression problem?
    hidden_layer_topology : type
        Hidden Layer topology, list of size number of hidden layers for MLP and each element contains number of neurons per layer.
    dropout_p : type
        Apply dropout regularization to reduce overfitting.
    add_softmax : type
        Softmax the output before evaluation.

    Attributes
    ----------
    vae : type
        Pytorch VAE module.
    topology : type
        List with hidden layer topology of MLP.
    mlp_layers : type
        All MLP layers (# layers and neurons per layer)
    output_layer : type
        nn.Linear connecting last MLP layer and output nodes.
    mlp : type
        nn.Sequential wraps all layers into sequential ordered pytorch module.
    output_z : type
        Whether to output latent embeddings.
    n_output
    categorical
    add_softmax
    dropout_p

    """

    # add ability to train decoderF
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
        """Pass data in to return predictions and embeddings.

        Parameters
        ----------
        x : type
            Input data.

        Returns
        -------
        torch.tensor
            Predictions
        torch.tensor
            Embeddings

        """
        z=self.vae.get_latent_z(x)
        return self.mlp(z), z

    def decode(self,z):
        """Run VAE decoder on embeddings.

        Parameters
        ----------
        z : type
            Embeddings.

        Returns
        -------
        torch.tensor
            Reconstructed Input.

        """
        return self.vae.decoder(z)

    def forward_embed(self,x):
        """Return predictions, latent embeddings and reconstructed input.

        Parameters
        ----------
        x : type
            Input data

        Returns
        -------
        torch.tensor
            Predictions
        torch.tensor
            Embeddings
        torch.tensor
            Reconstructed input.

        """
        out=self.vae.get_latent_z(x)
        recon=self.vae.decoder(out)
        return self.mlp(out), out, recon

    def toggle_latent_z(self):
        """Toggle whether to output latent embeddings during forward pass.        """
        if self.output_z:
            self.output_z=False
        else:
            self.output_z=True

    def forward_predict(self,x):
        """Make predictions, based on output_z, either output predictions or output embeddings.

        Parameters
        ----------
        x : type
            Input Data.

        Returns
        -------
        torch.tensor
            Predictions or embeddings.

        """
        if self.output_z:
            return self.vae.get_latent_z(x)
        else:
            return self.mlp(self.vae.get_latent_z(x))
