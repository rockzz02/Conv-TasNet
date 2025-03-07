import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchviz import make_dot
import numpy as np
import random
import soundfile as sf

from utility import models, sdr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_rate = 0

def load_wav_to_tensor(file_path):
    waveform, sample_rate = sf.read(file_path)
    waveform = torch.tensor(waveform).unsqueeze(0)
    waveform = waveform.to(torch.float32)
    return waveform, sample_rate

def save_tensor_to_wav(tensor, file_path, sample_rate=sample_rate):
    tensor = tensor.squeeze(0).numpy()
    sf.write(file_path, tensor, sample_rate)

def pad_waveform(waveform, target_length):
    current_length = waveform.shape[1]
    if current_length < target_length:
        padding = target_length - current_length
        waveform = F.pad(waveform, (0, padding), "constant", 0)
    return waveform

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
        
        # print(input.shape)
        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)
        # print(output.shape)
        # print(rest)
        
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

def si_snr_loss(source, estimate_source, eps=1e-8):
    """ Calculate SI-SNRi loss """
    def l2_norm(s):
        return torch.sqrt(torch.sum(s ** 2, dim=-1, keepdim=True) + eps)

    # Zero-mean norm
    source = source - torch.mean(source, dim=-1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)

    # SI-SNR
    s_target = torch.sum(source * estimate_source, dim=-1, keepdim=True) * source / l2_norm(source)
    e_noise = estimate_source - s_target
    si_snr = 20 * torch.log10(l2_norm(s_target) / (l2_norm(e_noise) + eps))
    return -torch.mean(si_snr)

def train_conv_tasnet(model, train_loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = si_snr_loss(targets, outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0
        
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}")
    print("Finished Training")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_conv_tasnet(model, test_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = si_snr_loss(targets, outputs)
            total_loss += loss.item()
            print(f"Batch {i + 1}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(test_loader)
    print(f"Average Loss: {avg_loss:.4f}")

    inputs, targets = next(iter(test_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    # print(outputs.shape)
    s1, s2 = torch.unbind(outputs, dim=1)
    s1 = s1.detach().cpu().numpy()
    s2 = s2.detach().cpu().numpy()
    x = inputs[0].detach().cpu().numpy()

    # Reshape the data for visualization
    s1 = s1.reshape(1, -1)
    s2 = s2.reshape(1, -1)
    x = x.reshape(1, -1)
    # save_tensor_to_wav(torch.tensor(s1), 'output1.wav')
    # save_tensor_to_wav(torch.tensor(s2), 'output2.wav')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(x, aspect='auto')
    axes[0].set_title('Input x')
    axes[0].set_xlim(0, 20)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(s1, aspect='auto')
    axes[1].set_title('Output y[0] (Speaker 1)')
    axes[1].set_xlim(0, 20)
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(s2, aspect='auto')
    axes[2].set_title('Output y[1] (Speaker 2)')
    axes[2].set_xlim(0, 20)
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

def main():
    set_seed(2)
    x_list = [torch.rand(1, 32000) for _ in range(100)]
    set_seed(3)
    y_list = [torch.rand(2, 32000) for _ in range(100)]

    # Stack the list into tensors
    x = torch.stack(x_list)
    y = torch.stack(y_list)

    # Create a DataLoader
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # # Initialize the model and optimizer
    model = TasNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_conv_tasnet(model, train_loader, optimizer, num_epochs=10)

    # Save the model
    torch.save(model.state_dict(), "tasnet_model.pth")


    set_seed(4)
    x_list = [torch.rand(1, 32000) for _ in range(10)]
    set_seed(5)
    y_list = [torch.rand(2, 32000) for _ in range(10)]

    # global sample_rate
    # wf1, sr1 = load_wav_to_tensor('male-male-mixture.wav')
    # wf2, sr2 = load_wav_to_tensor('male-male-dpcl1.wav')
    # wf3, sr3 = load_wav_to_tensor('male-male-dpcl2.wav')
    # sample_rate = sr3

    # print(wf1.shape)
    # print(wf2.shape)
    # print(wf3.shape)
    
    # max_length = max(wf1.shape[1], wf2.shape[1], wf3.shape[1])
    # wf1 = pad_waveform(wf1, max_length)
    # wf2 = pad_waveform(wf2, max_length)
    # wf3 = pad_waveform(wf3, max_length)

    # x_list = [wf1]
    # y_list = [torch.cat((wf2, wf3), dim=0)]

    # Stack the list into tensors
    x = torch.stack(x_list)
    y = torch.stack(y_list)

    # Create a DataLoader
    dataset = TensorDataset(x, y)
    test_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Test the model
    test_conv_tasnet(model, test_loader)

if __name__ == "__main__":
    main()



"""
1. Input -> Padding -> Encoder (self.encoder)
2. Encoder Output -> cLN (self.cLN)
3. cLN Output -> TCN (self.TCN)
4. TCN Output -> FCLayer (self.fc_layer)
5. FCLayer Output -> Mask Generation -> Masked Output
6. Masked Output -> Decoder (self.decoder)
7. Decoder Output -> Final Output
"""