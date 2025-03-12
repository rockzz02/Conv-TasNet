import torch
import random
import librosa
import numpy as np
import soundfile as sf
import torch.nn.functional as F

def load_wav_to_tensor(file_path):
    waveform, sample_rate = sf.read(file_path)
    waveform = torch.tensor(waveform).unsqueeze(0)
    waveform = waveform.to(torch.float32)
    return waveform, sample_rate

def save_tensor_to_wav(tensor, file_path, sample_rate):
    tensor = tensor.squeeze(0).numpy()
    sf.write(file_path, tensor, sample_rate)

def pad_waveform(waveform, target_length):
    current_length = waveform.shape[1]
    if current_length < target_length:
        padding = target_length - current_length
        waveform = F.pad(waveform, (0, padding), "constant", 0)
    return waveform

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def calculate_mse(audio_file1, audio_file2):
    y1, sr1 = librosa.load(audio_file1, sr=None)
    y2, sr2 = librosa.load(audio_file2, sr=None)

    print(type(y1))

    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1

    max_len = max(len(y1), len(y2))

    if len(y1) < max_len:
        y1 = np.pad(y1, (0, max_len - len(y1)))
    if len(y2) < max_len:
        y2 = np.pad(y2, (0, max_len - len(y2)))

    mse = np.mean((y1 - y2) ** 2)
    return mse