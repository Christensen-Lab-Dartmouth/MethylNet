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
