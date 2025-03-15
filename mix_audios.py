import os
import traceback
import numpy as np
import soundfile as sf

def audio_mix(a1, a2):
    try:
        audio1, sr1 = sf.read(a1)
        audio2, sr2 = sf.read(a2)

        if sr1 != sr2:
            # raise ValueError("Sample rates of the two audio files must be the same!")
            pass

        min_length = min(len(audio1), len(audio2))
        audio1 = audio1[:min_length]
        audio2 = audio2[:min_length]

        mixed_audio = audio1 + audio2
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))

        a1 = a1.split("/")[-1].split(".")[0]
        a2 = a2.split("/")[-1].split(".")[0]
        sf.write(f"audios/train_dir/mixed_audios/{a1}_{a2}.wav", mixed_audio, sr1)
    except Exception:
        print(traceback.format_exc())

files = os.listdir("audios/train_dir/clean_audios")
files = [i for i in files if i.find('.wav') != -1]

pth = lambda x: os.path.join("audios/train_dir/clean_audios", x)
for f1 in range(len(files)):
    for f2 in range(f1+1, len(files)):
        audio_mix(pth(files[f1]), pth(files[f2]))