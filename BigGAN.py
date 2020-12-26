import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

import layers
from sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}
  arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch


def shuffling(x):
    B,C,H,W = x.shape
    permute = torch.randperm(H*W)
    x = x.view(B,C,H*W)[:,:,permute]
    x = x.view(B,C,H,W)
    return x

# shared_dim=0
# hier=False
class Generator(nn.Module):
  def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
               G_kernel_size=3, G_attn='64', n_classes=1000,
               num_G_SVs=1, num_G_SV_itrs=1,
               G_shared=True, shared_dim=0, hier=False,
               cross_replica=False, mybn=False,
               G_activation=nn.ReLU(inplace=False),
               G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
               G_init='ortho', skip_init=False, no_optim=False,
               G_param='SN', norm_style='bn',
               **kwargs):
    super(Generator, self).__init__()
    # Channel width mulitplier
    self.ch = G_ch
    # Dimensionality of the latent space
    self.dim_z = dim_z
    # The initial spatial dimensions
    self.bottom_width = bottom_width
    # Resolution of the output
    self.resolution = resolution
    # Kernel size?
    self.kernel_size = G_kernel_size
    # Attention?
    self.attention = G_attn
    # number of classes, for use in categorical conditional generation
    # number of classes, for use in categorical conditional generation
    self.n_classes = n_classes
    # Use shared embeddings?
    self.G_shared = G_shared
    # Dimensionality of the shared embedding? Unused if not using G_shared
    self.shared_dim = shared_dim if shared_dim > 0 else dim_z
    # Hierarchical latent space?
    self.hier = hier
    # Cross replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # nonlinearity for residual blocks
    self.activation = G_activation
    # Initialization style
    self.init = G_init
    # Parameterization style
    self.G_param = G_param
    # Normalization style
    self.norm_style = norm_style
    # Epsilon for BatchNorm?
    self.BN_eps = BN_eps
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # fp16?
    self.fp16 = G_fp16
    # Architecture dict
    self.arch = G_arch(self.ch, self.attention)[resolution]

    # If using hierarchical latents, adjust z
    if self.hier:
      # Number of places z slots into
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.dim_z = self.z_chunk_size *  self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0

    # Which convs, batchnorms, and linear layers to use
    if self.G_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
      
    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.which_embedding = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                 else self.which_embedding)
    self.which_bn = functools.partial(layers.ccbn,
                          which_linear=bn_linear,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                      else self.n_classes),
                          norm_style=self.norm_style,
                          eps=self.BN_eps)


    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 
                    else layers.identity())
    # First linear layer
    self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    self.arch['in_channels'][0] * (self.bottom_width **2))

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation=self.activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None))]]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                cross_replica=self.cross_replica,
                                                mybn=self.mybn),
                                    self.activation,
                                    self.which_conv(self.arch['out_channels'][-1], 3))

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    # If this is an EMA copy, no need for an optim, so just return now
    if no_optim:
      return
    self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
    if G_mixed_precision:
      print('Using fp16 adam in G...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps)

    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  def forward(self, z, y, method = None, alpha = None, inx = None, inx2 = None):
    # If hierarchical, concatenate zs and ys
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)



    h = self.linear(z)



    if method == 'svd_linear':
        W = self.linear.weight
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        u, s, v = torch.svd(W)
        end.record()
        torch.cuda.synchronize()
        print("svd", start.elapsed_time(end))
        z[0,:] = z[0,:]+alpha*(v[:,inx])
        h = self.linear(z)


    # working
    if method == 'svd_gcircle':
        W = self.linear.weight.detach()
        u, s, v = torch.svd(W)
        Pv = v[:, inx].unsqueeze(0).T.matmul(v[:, inx].unsqueeze(0))
        Pvl = torch.eye(20).cuda() - Pv
        pvzzo = Pvl.matmul(z[0, :]).unsqueeze(0)
        z_norm = z.norm(2)
        step = alpha
        tetain = torch.sign(z.matmul(v[:, inx].unsqueeze(0).T))
        theta_zero = tetain*np.arccos((pvzzo.norm(2)/z_norm).detach().cpu().numpy())
        # z[0, :] = z_norm*((torch.cos((theta_zero+step)))*(pvzzo/pvzzo.norm(2))+(torch.sin(theta_zero+step))*v[:,inx])
        z[0, :] = z_norm*((torch.cos((theta_zero+step)))*(pvzzo/pvzzo.norm(2))+(torch.sin(theta_zero+step))*v[:,inx])

        print("stop when step = " , np.pi/2.0-theta_zero)
        print(torch.sin(theta_zero+step))
        h = self.linear(z)


    if method == 'svd_scircle':
        W = self.linear.weight
        u, s, v = torch.svd(W)
        vv = torch.zeros((20,2)).cuda()
        vv[:,0] = v[:,inx]
        vv[:,1] = v[:,inx2]
        Pv = vv.matmul(vv.T)
        Pvl = torch.eye(20).cuda() - Pv
        Pvz = (Pv.matmul(z[0, :])).T.matmul(Pv.matmul(z[0, :]))
        pvzzo = Pvl.matmul(z[0, :]).unsqueeze(0)
        theta = alpha

        v_ref = v[:,inx2:inx2+1]
        pv_ref = v_ref.matmul(v_ref.T)
        pv_refzo = pv_ref.matmul(z[0,:]).unsqueeze(0)
        tetain = torch.sign((Pv.matmul(z[0,:])).matmul(v[:,inx].unsqueeze(0).T))
        theta_zero = tetain * np.arccos((pv_refzo.norm(2) / torch.sqrt(Pvz)).detach().cpu().numpy())

        z[0, :] =  pvzzo+ torch.sqrt(Pvz)*(torch.cos(theta_zero+theta)*v[:,inx2] + torch.sin(theta_zero+theta)*v[:,inx])

        print(torch.sin(alpha))

        print("stop when cosine = 0 = " , np.pi/2.0-theta_zero)


        h = self.linear(z)



    if method == 'nl_shifty' and alpha > 0:
        P = torch.zeros((24576, 24576)).cuda()
        T = torch.eye(24576 - 4).cuda()
        P[0:24576 - 4, 4:] = T  # shift up
        D = torch.zeros((24576, 24576)).cuda()
        T = torch.zeros((24576)).cuda()
        ix = torch.arange(0, 24576, 16)
        for ixy in range(ix.shape[0]):
            T[ix[ixy]:ix[ixy] + 12] = 1.0
        D.as_strided([24576], [24576 + 1]).copy_(T)
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

        DWT = (D.matmul(W)).T
        DW = D.matmul(W)
        PW = D.matmul(P).matmul(W)

        M = torch.zeros(z.shape).cuda()
        for i in range(z.shape[1]):
            norma = DWT[i, :].matmul(DW[:, i])
            M[:, i] = (DWT[i, :].matmul(PW[:, i])) / norma

        M_n = M.pow(inx * alpha)

        if alpha == 0:
            z = z

        if alpha != 0:
            add_coeff = 0
            for i in range(1, 1 + alpha):
                add_coeff = add_coeff + (M.pow(inx*i))
            # add_coeff = 1 / (add_coeff)
            z = z * M_n + add_coeff * q
        h = self.linear(z)

    if method == 'nl_shifty' and alpha < 0:
        alpha = -alpha
        P = torch.zeros((24576, 24576)).cuda()
        T = torch.eye(24576 - 4).cuda()
        P[4:, 0:24576 - 4] = T  # shift up
        D = torch.zeros((24576, 24576)).cuda()
        T = torch.zeros((24576)).cuda()
        ix = torch.arange(4, 24576, 16)
        for ixy in range(ix.shape[0]):
            T[ix[ixy]:ix[ixy] + 12] = 1.0
        D.as_strided([24576], [24576 + 1]).copy_(T)
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

        DWT = (D.matmul(W)).T
        DW = D.matmul(W)
        PW = D.matmul(P).matmul(W)

        M = torch.zeros(z.shape).cuda()
        for i in range(z.shape[1]):
            norma = DWT[i, :].matmul(DW[:, i])
            M[:, i] = (DWT[i, :].matmul(PW[:, i])) / norma

        M_n = M.pow(inx * alpha)

        if alpha == 0:
            z = z

        if alpha != 0:
            add_coeff = 0
            for i in range(1, 1 + alpha):
                add_coeff = add_coeff + (M.pow(inx*i))
            # add_coeff = 1 / (add_coeff)
            z = z * M_n + add_coeff * q
        h = self.linear(z)


    # if method == 'nl_shifty' and alpha > 0:
    #     P = torch.zeros((24576, 24576)).cuda()
    #     T = torch.eye(24576 - 4).cuda()
    #     P[0:24576 - 4, 4:] = T  # shift up
    #     D = torch.zeros((24576, 24576)).cuda()
    #     T = torch.zeros((24576)).cuda()
    #     ix = torch.arange(0, 24576, 16)
    #     for ixy in range(ix.shape[0]):
    #         T[ix[ixy]:ix[ixy] + 12] = 1.0
    #     D.as_strided([24576], [24576 + 1]).copy_(T)
    #     W = self.linear.weight.detach()
    #     b = self.linear.bias.detach()
    #     l = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
    #     DWT = (D.matmul(W)).T
    #     DW = D.matmul(W)
    #     PW = P.matmul(W)
    #
    #     q = torch.zeros(z.shape).cuda()
    #     for i in range(z.shape[1]):
    #         norma = DWT[i, :].matmul(DW[:, i])
    #         q[:, i] = (DWT[i, :].matmul(PW[:, i])) / norma
    #     M_n = q.pow(inx*alpha)
    #
    #     if alpha == 0:
    #         z = z
    #
    #     if alpha != 0:
    #         add_coeff = 0
    #         for i in range(1,1+alpha):
    #             add_coeff = add_coeff + (q.pow(inx*i*alpha))
    #         add_coeff = 1/(add_coeff)
    #         z = z*M_n + add_coeff*l
    #     h = self.linear(z)
    #
    #
    #
    # if method == 'nl_shifty' and alpha < 0:
    #     alpha = -alpha
    #     P = torch.zeros((24576, 24576)).cuda()
    #     T = torch.eye(24576 - 4).cuda()
    #     P[4:, 0:24576 - 4] = T  # shift up
    #     D = torch.zeros((24576, 24576)).cuda()
    #     T = torch.zeros((24576)).cuda()
    #     ix = torch.arange(4, 24576, 16)
    #     for ixy in range(ix.shape[0]):
    #         T[ix[ixy]:ix[ixy] + 12] = 1.0
    #     D.as_strided([24576], [24576 + 1]).copy_(T)
    #     W = self.linear.weight.detach()
    #     b = self.linear.bias.detach()
    #     l = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
    #     DWT = (D.matmul(W)).T
    #     DW = D.matmul(W)
    #     PW = P.matmul(W)
    #     q = torch.zeros(z.shape).cuda()
    #     for i in range(z.shape[1]):
    #         norma = DWT[i, :].matmul(DW[:, i])
    #         q[:, i] = (DWT[i, :].matmul(PW[:, i])) / norma
    #     M_n = q.pow(inx*alpha)
    #     if alpha == 0:
    #         z = z
    #     if alpha != 0:
    #         add_coeff = 0
    #         for i in range(1,1+(alpha)):
    #             add_coeff = add_coeff + (q.pow(inx*i*alpha))
    #         add_coeff = 1/(add_coeff)
    #         z = z*M_n + add_coeff*l
    #     h = self.linear(z)



        if method == 'gcircle_shifty' and alpha > 0:
            P = torch.zeros((24576, 24576)).cuda()
            T = torch.eye(24576 - 4).cuda()
            P[0:24576 - 4, 4:] = T  # shift up
            D = torch.zeros((24576, 24576)).cuda()
            T = torch.zeros((24576)).cuda()
            ix = torch.arange(0, 24576, 16)
            for ixy in range(ix.shape[0]):
                T[ix[ixy]:ix[ixy] + 12] = 1.0
            D.as_strided([24576], [24576 + 1]).copy_(T)
            W = self.linear.weight.detach()
            b = self.linear.bias.detach()
            q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

            Pv = q.unsqueeze(0).T.matmul(q.unsqueeze(0)).cuda()
            Pvl = torch.eye(20).cuda() - Pv.cuda()
            Pvz = (Pv.matmul(z[0, :])).T.matmul(Pv.matmul(z[0, :])).cuda()
            pvzzo = Pvl.matmul(z[0, :]).unsqueeze(0).cuda()
            z_norm = z.norm(2)
            step = alpha  # /z_norm
            D = torch.sign(z.matmul(q.unsqueeze(0).T.cuda()))
            theta_zero = D * np.arccos((pvzzo.norm(2) / z_norm).detach().cpu().numpy())
            z[0, :] = z_norm * ((torch.cos((theta_zero + step))) * (pvzzo / pvzzo.norm(2)) + (
                torch.sin(theta_zero + step)) * q.cuda())
            h = self.linear(z)
            h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)


    if method == 'gcircle_shifty' and alpha < 0:
        alpha = -alpha
        P = torch.zeros((24576, 24576)).cuda()
        T = torch.eye(24576 - 4).cuda()
        P[4:, 0:24576 - 4] = T  # shift up
        D = torch.zeros((24576, 24576)).cuda()
        T = torch.zeros((24576)).cuda()
        ix = torch.arange(4, 24576, 16)
        for ixy in range(ix.shape[0]):
            T[ix[ixy]:ix[ixy] + 12] = 1.0
        D.as_strided([24576], [24576 + 1]).copy_(T)
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
        Pv = q.unsqueeze(0).T.matmul(q.unsqueeze(0)).cuda()
        Pvl = torch.eye(20).cuda() - Pv.cuda()
        pvzzo = Pvl.matmul(z[0, :]).unsqueeze(0).cuda()
        z_norm = z.norm(2)
        step = alpha  # /z_norm
        D = torch.sign(z.matmul(q.unsqueeze(0).T.cuda()))
        theta_zero = D * np.arccos((pvzzo.norm(2) / z_norm).detach().cpu().numpy())
        z[0, :] = z_norm * ((torch.cos((theta_zero + step))) * (pvzzo / pvzzo.norm(2)) + (
            torch.sin(theta_zero + step)) * q.cuda())
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

    # if method == 1 and inx == -1000 and alpha > 0:
    #     P = torch.zeros((24576, 24576)).cuda()
    #     T = torch.eye(24576-4).cuda()
    #     P[0:24576-4,4:] = T # shift up
    #     D = torch.zeros((24576, 24576)).cuda()
    #     T = torch.zeros((24576)).cuda()
    #     ix = torch.arange(0, 24576, 16)
    #     for ixy in range(ix.shape[0]):
    #         T[ix[ixy]:ix[ixy] + 12] = 1.0
    #     D.as_strided([24576], [24576 + 1]).copy_(T)
    #     W = self.linear.weight.detach()
    #     b = self.linear.bias.detach()
    #     q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
    #     Pv = q.unsqueeze(0).T.matmul(q.unsqueeze(0)).cuda()
    #     Pvl = torch.eye(20).cuda() - Pv.cuda()
    #     pvzzo = Pvl.matmul(z[0, :]).unsqueeze(0).cuda()
    #     z_norm = z.norm(2)
    #     step = alpha  # /z_norm
    #     D = torch.sign(z.matmul(q.unsqueeze(0).T.cuda()))
    #     theta_zero = D * np.arccos((pvzzo.norm(2) / z_norm).detach().cpu().numpy())
    #     z[0, :] = z_norm * ((torch.cos((theta_zero + step))) * (pvzzo / pvzzo.norm(2)) + (
    #         torch.sin(theta_zero + step)) * q.cuda())
    #     h = self.linear(z)
    #     h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

    if method == 'l_shifty' and alpha > 0:
        P = torch.zeros((24576, 24576)).cuda()
        T = torch.eye(24576-4).cuda()
        P[0:24576-4,4:] = T # shift up
        D = torch.zeros((24576, 24576)).cuda()
        T = torch.zeros((24576)).cuda()
        ix = torch.arange(0, 24576, 16)
        for ixy in range(ix.shape[0]):
            T[ix[ixy]:ix[ixy] + 12] = 1.0
        D.as_strided([24576], [24576 + 1]).copy_(T)
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
        end.record()
        torch.cuda.synchronize()
        z = z + alpha * q
        h = self.linear(z)

    if method == 'l_shifty' and alpha < 0:
        alpha = -alpha
        P = torch.zeros((24576, 24576)).cuda()
        T = torch.eye(24576-4).cuda()
        P[4:,0:24576-4] = T # shift up
        D = torch.zeros((24576, 24576)).cuda()
        T = torch.zeros((24576)).cuda()
        ix = torch.arange(4, 24576, 16)
        for ixy in range(ix.shape[0]):
            T[ix[ixy]:ix[ixy] + 12] = 1.0
        D.as_strided([24576], [24576 + 1]).copy_(T)
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
        z = z + alpha * q
        h = self.linear(z)

    if method == 'nl_shiftx' and alpha > 0:
        P = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([0, 1, 1, 1])).repeat(6144).cuda()
        P.as_strided([24576], [24576 + 1]).copy_(T)
        P = torch.roll(P, -1, [1])
        P = P.T
        D = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([1, 1, 1, 0])).repeat(6144).cuda()
        D.as_strided([24576], [24576 + 1]).copy_(T)
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
        DWT = (D.matmul(W)).T
        DW = D.matmul(W)

        PW = D.matmul(P).matmul(W)
        M = torch.zeros(z.shape).cuda()
        for i in range(z.shape[1]):
            norma = DWT[i, :].matmul(DW[:, i])
            M[:, i] = (DWT[i, :].matmul(PW[:, i])) / norma

        M_n = M.pow(inx* alpha)

        if alpha == 0:
            z = z
        if alpha != 0:
            add_coeff = 0
            for i in range(1, 1 + alpha):
                add_coeff = add_coeff + (M.pow(inx * i))
            # add_coeff = 1 / (add_coeff)
            z = z * M_n + add_coeff * q
        h = self.linear(z)


    if method =='nl_shiftx' and alpha < 0:
        alpha = -alpha
        P = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([1, 1, 1, 0])).repeat(6144).cuda()
        P.as_strided([24576], [24576 + 1]).copy_(T)
        P = torch.roll(P, 1, [1])
        P = P.T
        D = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([0, 1, 1, 1])).repeat(6144).cuda()
        D.as_strided([24576], [24576 + 1]).copy_(T)
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
        DWT = (D.matmul(W)).T
        DW = D.matmul(W)
        PW = D.matmul(P).matmul(W)
        M = torch.zeros(z.shape).cuda()
        for i in range(z.shape[1]):
            norma = DWT[i, :].matmul(DW[:, i])
            M[:, i] = (DWT[i, :].matmul(PW[:, i])) / norma

        M_n = M.pow(inx * alpha)

        if alpha == 0:
            z = z
        if alpha != 0:
            add_coeff = 0
            for i in range(1, 1 + alpha):
                add_coeff = add_coeff + (M.pow(inx * i))
            # add_coeff = 1 / (add_coeff)
            z = z * M_n + add_coeff * q
        h = self.linear(z)

    # if method == 'nl_shiftx' and alpha > 0:
    #     P = torch.zeros((24576, 24576)).cuda()
    #     T = (torch.tensor([0, 1, 1, 1])).repeat(6144).cuda()
    #     P.as_strided([24576], [24576 + 1]).copy_(T)
    #     P = torch.roll(P, -1, [1])
    #     P = P.T
    #     D = torch.zeros((24576, 24576)).cuda()
    #     T = (torch.tensor([1, 1, 1, 0])).repeat(6144).cuda()
    #     D.as_strided([24576], [24576 + 1]).copy_(T)
    #     W = self.linear.weight.detach()
    #     b = self.linear.bias.detach()
    #     l = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
    #     DWT = (D.matmul(W)).T
    #     DW = D.matmul(W)
    #     PW = P.matmul(W)
    #     q = torch.zeros(z.shape).cuda()
    #     for i in range(z.shape[1]):
    #         norma = DWT[i, :].matmul(DW[:, i])
    #         q[:, i] = (DWT[i, :].matmul(PW[:, i])) / norma
    #     M_n = q.pow(inx* alpha)
    #     if alpha == 0:
    #         z = z
    #     if alpha != 0:
    #         add_coeff = 0
    #         for i in range(1, 1 + alpha):
    #             add_coeff = add_coeff + (q.pow(inx * i * alpha))
    #         add_coeff = 1 / (add_coeff)
    #         z = z * M_n + add_coeff * l
    #     h = self.linear(z)
    # if method =='nl_shiftx' and alpha < 0:
    #     alpha = -alpha
    #     P = torch.zeros((24576, 24576)).cuda()
    #     T = (torch.tensor([1, 1, 1, 0])).repeat(6144).cuda()
    #     P.as_strided([24576], [24576 + 1]).copy_(T)
    #     P = torch.roll(P, 1, [1])
    #     P = P.T
    #     D = torch.zeros((24576, 24576)).cuda()
    #     T = (torch.tensor([0, 1, 1, 1])).repeat(6144).cuda()
    #     D.as_strided([24576], [24576 + 1]).copy_(T)
    #     W = self.linear.weight.detach()
    #     b = self.linear.bias.detach()
    #     l = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
    #     DWT = (D.matmul(W)).T
    #     DW = D.matmul(W)
    #     PW = P.matmul(W)
    #     q = torch.zeros(z.shape).cuda()
    #     for i in range(z.shape[1]):
    #         norma = DWT[i, :].matmul(DW[:, i])
    #         q[:, i] = (DWT[i, :].matmul(PW[:, i])) / norma
    #
    #     M_n = q.pow(inx * alpha)
    #     if alpha == 0:
    #         z = z
    #     if alpha != 0:
    #         add_coeff = 0
    #         for i in range(1, 1 + alpha):
    #             add_coeff = add_coeff + (q.pow(inx* i * alpha))
    #         add_coeff = 1 / (add_coeff)
    #         z = z * M_n + add_coeff * l
    #     h = self.linear(z)

    if method == 'gcircle_shiftx' and alpha > 0:
        # additive mode
        P = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([0, 1, 1, 1])).repeat(6144).cuda()
        P.as_strided([24576], [24576 + 1]).copy_(T)
        P = torch.roll(P, -1, [1])
        P = P.T
        D = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([1, 1, 1, 0])).repeat(6144).cuda()
        D.as_strided([24576], [24576 + 1]).copy_(T)
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

        Pv = q.unsqueeze(0).T.matmul(q.unsqueeze(0)).cuda()
        Pvl = torch.eye(20).cuda() - Pv.cuda()
        Pvz = (Pv.matmul(z[0, :])).T.matmul(Pv.matmul(z[0, :])).cuda()
        pvzzo = Pvl.matmul(z[0, :]).unsqueeze(0).cuda()
        z_norm = z.norm(2)
        step = alpha  # /z_norm
        D = torch.sign(z.matmul(q.unsqueeze(0).T.cuda()))
        theta_zero = D * np.arccos((pvzzo.norm(2) / z_norm).detach().cpu().numpy())
        z[0, :] = z_norm * ((torch.cos((theta_zero + step))) * (pvzzo / pvzzo.norm(2)) + (torch.sin(theta_zero + step)) * q.cuda())
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

    if method == 'gcircle_shiftx' and inx == -1000 and alpha < 0:
        # additive mode
        alpha = -alpha
        P = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([1, 1, 1, 0])).repeat(6144).cuda()
        P.as_strided([24576], [24576 + 1]).copy_(T)
        P = torch.roll(P, 1, [1])
        P = P.T
        D = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([0, 1, 1, 1])).repeat(6144).cuda()
        D.as_strided([24576], [24576 + 1]).copy_(T)
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

        Pv = q.unsqueeze(0).T.matmul(q.unsqueeze(0)).cuda()
        Pvl = torch.eye(20).cuda() - Pv.cuda()
        Pvz = (Pv.matmul(z[0, :])).T.matmul(Pv.matmul(z[0, :])).cuda()
        pvzzo = Pvl.matmul(z[0, :]).unsqueeze(0).cuda()
        z_norm = z.norm(2)
        step = alpha  # /z_norm
        D = torch.sign(z.matmul(q.unsqueeze(0).T.cuda()))
        theta_zero = D * np.arccos((pvzzo.norm(2) / z_norm).detach().cpu().numpy())
        z[0, :] = z_norm * ((torch.cos((theta_zero + step))) * (pvzzo / pvzzo.norm(2)) + (
            torch.sin(theta_zero + step)) * q.cuda())
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

    if method == 'l_shiftx' and alpha > 0:
        P = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([0, 1, 1, 1])).repeat(6144).cuda()
        P.as_strided([24576], [24576 + 1]).copy_(T)
        P = torch.roll(P, -1, [1])
        P = P.T
        D = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([1,1,1,0])).repeat(6144).cuda()
        D.as_strided([24576], [24576 + 1]).copy_(T)
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
        z = z + alpha * q
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

    if method == 'l_shiftx' and alpha < 0:
        alpha = -alpha
        P = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([1, 1, 1, 0])).repeat(6144).cuda()
        P.as_strided([24576], [24576 + 1]).copy_(T)
        P = torch.roll(P, 1, [1])
        P = P.T
        D = torch.zeros((24576, 24576)).cuda()
        T = (torch.tensor([0,1,1,1])).repeat(6144).cuda()
        D.as_strided([24576], [24576 + 1]).copy_(T)
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
        z = z + alpha * q
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)


    if method == 'nl_zoom' and alpha > 0:
        P = torch.zeros((24576, 24576)).cuda()
        T = torch.tensor([0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0]).repeat(1536).cuda()
        P.as_strided([24576], [24576 + 1]).copy_(T)
        D = torch.eye(24576).cuda()
        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
        DWT = (D.matmul(W)).T
        DW = D.matmul(W)
        PW = D.matmul(P).matmul(W)

        M = torch.zeros(z.shape).cuda()
        for i in range(z.shape[1]):
            norma = DWT[i, :].matmul(DW[:, i])
            M[:, i] = (DWT[i, :].matmul(PW[:, i])) / norma

        M_n = M.pow(inx * alpha)

        if alpha == 0:
            z = z

        if alpha != 0:
            add_coeff = 0
            for i in range(1, 1 + alpha):
                add_coeff = add_coeff + (M.pow(inx * i))
            # add_coeff = 1 / (add_coeff)
            z = z * M_n + add_coeff * q
        h = self.linear(z)


    if method == 'nl_zoom' and alpha < 0:
        alpha = -alpha
        P = torch.eye(24576).cuda()

        D = torch.zeros((24576, 24576)).cuda()
        ix = torch.arange(5, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy], ix[ixy]] = 1.0
            D[ix[ixy] - 1, ix[ixy]] = 1.0
            D[ix[ixy] - 4, ix[ixy]] = 1.0
            D[ix[ixy] - 5, ix[ixy]] = 1.0

        ix = torch.arange(6, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy], ix[ixy]] = 1.0
            D[ix[ixy] - 4, ix[ixy]] = 1.0
            D[ix[ixy] - 3, ix[ixy]] = 1.0
            D[ix[ixy] + 1, ix[ixy]] = 1.0

        ix = torch.arange(9, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy], ix[ixy]] = 1.0
            D[ix[ixy] - 1, ix[ixy]] = 1.0
            D[ix[ixy] + 3, ix[ixy]] = 1.0
            D[ix[ixy] + 4, ix[ixy]] = 1.0

        ix = torch.arange(10, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy], ix[ixy]] = 1.0
            D[ix[ixy] + 1, ix[ixy]] = 1.0
            D[ix[ixy] + 5, ix[ixy]] = 1.0
            D[ix[ixy] + 4, ix[ixy]] = 1.0

        W = self.linear.weight.detach()
        b = self.linear.bias.detach()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

        DWT = (D.matmul(W)).T
        DW = D.matmul(W)
        PW = D.matmul(P).matmul(W)

        M = torch.zeros(z.shape).cuda()
        for i in range(z.shape[1]):
            norma = DWT[i, :].matmul(DW[:, i])
            M[:, i] = (DWT[i, :].matmul(PW[:, i])) / norma

        M_n = M.pow(inx * alpha)

        if alpha == 0:
            z = z

        if alpha != 0:
            add_coeff = 0
            for i in range(1, 1 + alpha):
                add_coeff = add_coeff + (M.pow(inx * i))
            # add_coeff = 1 / (add_coeff)
            z = z * M_n + add_coeff * q
        h = self.linear(z)


    if method == 'l_zoom' and alpha > 0:
        # additive mode
        P = torch.zeros((24576, 24576))
        T = torch.tensor([0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0]).repeat(1536)
        P.as_strided([24576], [24576 + 1]).copy_(T)
        D = torch.eye(24576)
        W = self.linear.weight.detach().cpu()
        b = self.linear.bias.detach().cpu()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))
        z = z + alpha * q.cuda()
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)




    if method == 'gcircle_zoom' and alpha > 0:
        # additive mode
        P = torch.zeros((24576, 24576))
        T = torch.tensor([0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0]).repeat(1536)
        P.as_strided([24576], [24576 + 1]).copy_(T)
        D = torch.eye(24576)
        W = self.linear.weight.detach().cpu()
        b = self.linear.bias.detach().cpu()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b)).cuda().T
        Pv = q.unsqueeze(0).T.matmul(q.unsqueeze(0)).cuda()
        Pvl = torch.eye(20).cuda() - Pv.cuda()
        pvzzo = Pvl.matmul(z[0, :]).unsqueeze(0).cuda()
        z_norm = z.norm(2)
        step = alpha
        D = torch.sign(z.matmul(q.unsqueeze(0).T.cuda()))
        theta_zero = D * np.arccos((pvzzo.norm(2) / z_norm).detach().cpu().numpy())
        z[0, :] = z_norm * ((torch.cos((theta_zero + step))) * (pvzzo / pvzzo.norm(2)) + (torch.sin(theta_zero + step)) * q.cuda())
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)



    if method == 'gcircle_shiftx' and alpha > 0:
        # additive mode
        alpha = -alpha
        P = torch.eye(24576)
        D = torch.zeros((24576, 24576))
        ix = torch.arange(5, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy], ix[ixy]] = 1.0
            D[ix[ixy] - 1, ix[ixy]] = 1.0
            D[ix[ixy] - 4, ix[ixy]] = 1.0
            D[ix[ixy] - 5, ix[ixy]] = 1.0

        ix = torch.arange(6, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy], ix[ixy]] = 1.0
            D[ix[ixy] - 4, ix[ixy]] = 1.0
            D[ix[ixy] - 3, ix[ixy]] = 1.0
            D[ix[ixy] + 1, ix[ixy]] = 1.0

        ix = torch.arange(9, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy], ix[ixy]] = 1.0
            D[ix[ixy] - 1, ix[ixy]] = 1.0
            D[ix[ixy] + 3, ix[ixy]] = 1.0
            D[ix[ixy] + 4, ix[ixy]] = 1.0

        ix = torch.arange(10, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy], ix[ixy]] = 1.0
            D[ix[ixy] + 1, ix[ixy]] = 1.0
            D[ix[ixy] + 5, ix[ixy]] = 1.0
            D[ix[ixy] + 4, ix[ixy]] = 1.0

        W = self.linear.weight.detach().cpu()
        b = self.linear.bias.detach().cpu()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b)).cuda().T

        Pv = q.unsqueeze(0).T.matmul(q.unsqueeze(0)).cuda()
        Pvl = torch.eye(20).cuda() - Pv.cuda()
        pvzzo = Pvl.matmul(z[0, :]).unsqueeze(0).cuda()
        z_norm = z.norm(2)
        step = alpha
        D = torch.sign(z.matmul(q.unsqueeze(0).T.cuda()))
        theta_zero = D * np.arccos((pvzzo.norm(2) / z_norm).detach().cpu().numpy())
        z[0, :] = z_norm * ((torch.cos((theta_zero + step))) * (pvzzo / pvzzo.norm(2)) + (torch.sin(theta_zero + step)) * q.cuda())
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)



    if method == 'l_zoom' and alpha < 0:
        alpha = -alpha

        P = torch.eye(24576)
        D = torch.zeros((24576, 24576))
        ix = torch.arange(5, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy],ix[ixy]] = 1.0
            D[ix[ixy]- 1,ix[ixy]] = 1.0
            D[ix[ixy] - 4,ix[ixy]] = 1.0
            D[ix[ixy] - 5,ix[ixy]] = 1.0

        ix = torch.arange(6, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy],ix[ixy]] = 1.0
            D[ix[ixy] - 4,ix[ixy]] = 1.0
            D[ix[ixy] - 3,ix[ixy]] = 1.0
            D[ix[ixy] + 1,ix[ixy]] = 1.0

        ix = torch.arange(9, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy],ix[ixy]] = 1.0
            D[ix[ixy]- 1,ix[ixy]] = 1.0
            D[ix[ixy] + 3, ix[ixy]] = 1.0
            D[ix[ixy] + 4, ix[ixy]] = 1.0

        ix = torch.arange(10, 24576, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy],ix[ixy]] = 1.0
            D[ix[ixy] + 1,ix[ixy]] = 1.0
            D[ix[ixy] + 5, ix[ixy]] = 1.0
            D[ix[ixy] + 4, ix[ixy]] = 1.0


        W = self.linear.weight.detach().cpu()
        b = self.linear.bias.detach().cpu()
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

        z = z + alpha * q.cuda()
        h = self.linear(z)

    # Reshape
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)


    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        if h.shape[2] == inx2:
            h = block(h, ys[index],alpha,inx, method = method)
        else:
            h = block(h, ys[index], alpha=0, inx=0, method = None)

    # Apply batchnorm-relu-conv-tanh at output
    return torch.tanh(self.output_layer(h))


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64',ksize='333333', dilation='111111'):
  arch = {}
  arch[256] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[128] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[64]  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}
  arch[32]  = {'in_channels' :  [3] + [item * ch for item in [4, 4, 4]],
               'out_channels' : [item * ch for item in [4, 4, 4, 4]],
               'downsample' : [True, True, False, False],
               'resolution' : [16, 16, 16, 16],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,6)}}
  return arch

class Discriminator(nn.Module):

  def __init__(self, D_ch=64, D_wide=True, resolution=128,
               D_kernel_size=3, D_attn='64', n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
               SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
               D_init='ortho', skip_init=False, D_param='SN', **kwargs):
    super(Discriminator, self).__init__()
    # Width multiplier
    self.ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = D_wide
    # Resolution
    self.resolution = resolution
    # Kernel size
    self.kernel_size = D_kernel_size
    # Attention?
    self.attention = D_attn
    # Number of classes
    self.n_classes = n_classes
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init = D_init
    # Parameterization style
    self.D_param = D_param
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # Fp16?
    self.fp16 = D_fp16
    # Architecture
    self.arch = D_arch(self.ch, self.attention)[resolution]

    # Which convs, batchnorms, and linear layers to use
    # No option to turn off SN in D right now
    if self.D_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_embedding = functools.partial(layers.SNEmbedding,
                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                              eps=self.SN_eps)
    # Prepare model
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                             self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
    # Embedding for projection discrimination
    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

    # Initialize weights
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
    if D_mixed_precision:
      print('Using fp16 adam in D...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, y=None):
    # Stick x into h for cleaner for loops without flow control
    h = x
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])
    # Get initial class-unconditional output
    out = self.linear(h)
    # Get projection of final featureset onto class vectors and add to evidence
    out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
    return out

# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
  def __init__(self, G, D):
    super(G_D, self).__init__()
    self.G = G
    self.D = D

  def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
              split_D=False):              
    # If training G, enable grad tape
    with torch.set_grad_enabled(train_G):
      # Get Generator output given noise
      G_z = self.G(z, self.G.shared(gy))
      # Cast as necessary
      if self.G.fp16 and not self.D.fp16:
        G_z = G_z.float()
      if self.D.fp16 and not self.G.fp16:
        G_z = G_z.half()
    # Split_D means to run D once with real data and once with fake,
    # rather than concatenating along the batch dimension.
    if split_D:
      D_fake = self.D(G_z, gy)
      if x is not None:
        D_real = self.D(x, dy)
        return D_fake, D_real
      else:
        if return_G_z:
          return D_fake, G_z
        else:
          return D_fake
    # If real data is provided, concatenate it with the Generator's output
    # along the batch dimension for improved efficiency.
    else:
      D_input = torch.cat([G_z, x], 0) if x is not None else G_z
      D_class = torch.cat([gy, dy], 0) if dy is not None else gy
      # Get Discriminator output
      D_out = self.D(D_input, D_class)
      if x is not None:
        return torch.split(D_out, [G_z.shape[0], x.shape[0]]) # D_fake, D_real
      else:
        if return_G_z:
          return D_out, G_z
        else:
          return D_out
