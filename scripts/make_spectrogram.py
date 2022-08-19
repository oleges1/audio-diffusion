import torchaudio
import librosa 
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


def main():
    filename = 'wavs/0.wav'
    audio, sample_rate_clean = torchaudio.load(filename)
    #audio = torch.FloatTensor(audio)
    speca = librosa.stft(audio.cpu().numpy())
    # speca = img[j].cpu().numpy()
    plt.figure(figsize=(10, 10))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(speca[0]), ref=np.max))
    plt.colorbar()
    plt.savefig(f'speca.png')
    # plt.close()
    # plt.imshow(img)
    # plt.show()
    
if __name__ == "__main__":
    main()
