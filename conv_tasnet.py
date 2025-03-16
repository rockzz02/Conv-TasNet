import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import matplotlib.pyplot as plt

from utility import models, sdr
from utils import load_wav_to_tensor, save_tensor_to_wav, adjust_length, set_seed, si_snr_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def test_conv_tasnet(model, test_loader, sr):
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
    outputs = torch.unbind(outputs, dim=1)
    outputs = [i.detach().cpu().numpy().reshape(1, -1) for i in outputs]
    x = inputs[0].detach().cpu().numpy()

    fig, axes = plt.subplots(len(outputs)+1, 1, figsize=(15, 8))
    x = x.reshape(1, -1)
    save_tensor_to_wav(torch.tensor(x), 'audios/input.wav', sr)
    axes[0].plot(x[0])
    axes[0].set_title('Input Mixture Waveform')
    axes[0].set_xlim(0, len(x[0]))
    axes[0].set_ylabel('Amplitude')
    
    for i, s in enumerate(outputs):
        save_tensor_to_wav(torch.tensor(s), f'audios/output{i+1}.wav', sr)

        axes[i+1].plot(s[0])
        axes[i+1].set_title(f'Output Speaker {i+1} Waveform')
        axes[i+1].set_xlim(0, len(s[0]))
        axes[i+1].set_ylabel('Amplitude')
    
    axes[len(outputs)].set_xlabel('Samples')
    
    plt.tight_layout()
    plt.show()

def main():
    # set_seed(2)
    # x_tr_list = [torch.rand(1, 32000) for _ in range(200)]
    # set_seed(3)
    # y_tr_list = [torch.rand(2, 32000) for _ in range(200)]

    files_in = os.listdir("audios/train_dir/mixed_audios")
    files_in = [i for i in files_in if i.find('.wav') != -1]
    x_tr_list = [load_wav_to_tensor(f'audios/train_dir/mixed_audios/{i}')[0] for i in files_in]
    # max_length = max([i.shape[1] for i in x_tr_list])
    # max_length = max(max_length, max([load_wav_to_tensor(f'audios/train_dir/clean_audios/{i}')[0].shape[1] for i in os.listdir("audios/train_dir/clean_audios") if i.find('.wav') != -1]))

    max_length = 60000
    x_tr_list = [adjust_length(i, max_length) for i in x_tr_list]
    
    y_tr_list = []
    for file in files_in:
        files_out = file.split('.')[0].split('_')
        files_out = [f'audios/train_dir/clean_audios/{i}.wav' for i in files_out]
        sep_y = [load_wav_to_tensor(i)[0] for i in files_out]
        sep_y = [adjust_length(i, max_length) for i in sep_y]
        # print(sep_y[0].shape)
        # print(sep_y[1].shape)
        sep_y = torch.cat(sep_y, dim=0)
        y_tr_list.append(sep_y)

    x_tr = torch.stack(x_tr_list)
    y_tr = torch.stack(y_tr_list)

    dataset = TensorDataset(x_tr, y_tr)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = TasNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_conv_tasnet(model, train_loader, optimizer, num_epochs=10)

    torch.save(model.state_dict(), "tasnet_model.pth")

    wf1, sr1 = load_wav_to_tensor('audios/male-male-mixture.wav')
    wf2, sr2 = load_wav_to_tensor('audios/male-male-dpcl1.wav')
    wf3, sr3 = load_wav_to_tensor('audios/male-male-dpcl2.wav')

    # print(wf1.shape)
    # print(wf2.shape)
    # print(wf3.shape)
    
    # max_length = max(wf1.shape[1], wf2.shape[1], wf3.shape[1])
    wf1 = adjust_length(wf1, max_length)
    wf2 = adjust_length(wf2, max_length)
    wf3 = adjust_length(wf3, max_length)

    x_ts_list = [wf1]
    y_ts_list = [torch.cat((wf2, wf3), dim=0)]

    x_ts = torch.stack(x_ts_list)
    y_ts = torch.stack(y_ts_list)

    dataset = TensorDataset(x_ts, y_ts)
    test_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    test_conv_tasnet(model, test_loader, sr3)

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