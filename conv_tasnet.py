import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

from utility import models, sdr


# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, win=2, layer=8, stack=3, 
                 kernel=3, num_spk=2, causal=False):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        
        self.win = int(sr*win/1000)
        self.stride = self.win // 2
        
        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal
        
        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        
        # TCN separator
        # Temporal Convolutional Network (TCN)
        self.TCN = models.TCN(self.enc_dim, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*4,
                              self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)
        # nn.ConvTranspose1d is a powerful tool in PyTorch for tasks that require upsampling and reconstruction of 1-dimensional data, making it an essential component in various neural network architectures for audio and signal processing.


    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        # print(input.dim())
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input):
        
        print(input.shape)
        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)
        print(output.shape)
        print(rest)
        
        # waveform encoder
        enc_output = self.encoder(output)  # B, N, L
        # B=Batch Size, N=Number of Channels, L=Length of Signal, C=Number of Speakers
        
        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L
        
        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T
        
        return output


def test_conv_tasnet():
    x1 = torch.rand(2, 32000)
    
    nnet = TasNet()
    
    x = nnet(x1)
    
    s1 = x[0].detach()
    s2 = x[1].detach()

    print(s1.shape)
    print(s2.shape)
    print(s1+s2 == x1.detach())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(x1.numpy(), aspect='auto')
    axes[0].set_title('Input x1')
    axes[0].set_xlim(0, 10)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(s1.numpy(), aspect='auto')
    axes[1].set_title('Output x[0] (Speaker 1)')
    axes[1].set_xlim(0, 10)
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(s2.numpy(), aspect='auto')
    axes[2].set_title('Output x[1] (Speaker 2)')
    axes[2].set_xlim(0, 10)
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_conv_tasnet()