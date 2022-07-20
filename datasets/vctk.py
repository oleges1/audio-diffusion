import torch
from torch.utils import data
from torchaudio import datasets
import torchaudio.functional as AF
from librosa.filters import mel as librosa_mel_fn
from einops import rearrange

hann_window = {}

def spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    if str(y.device) not in hann_window:
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    return spec

class VCTK(datasets.DR_VCTK):
    def __init__(
            self, root, segment_size, sampling_rate, n_fft, 
            num_mels, hop_size, win_size,
            subset='train', zero_out_percent=None
        ):
        super().__init__(self, root, subset, download=True)
        self.subset = subset
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.zero_out_percent = zero_out_percent

    
    def __getitem__(self, i):
        file_clean_audio = self._clean_audio_dir / self._filename_list[i]
        audio, sample_rate_clean = torchaudio.load(file_clean_audio)

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        if audio.size(1) >= self.segment_size:
            max_audio_start = audio.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start+self.segment_size]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
        
        speca = spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size,
                                   center=False)
        
        if self.zero_out_percent is not None:
            speca[:, speca.shape[1] // 2:] = 0
        
        speca = rearrange(speca, 'B S T D -> B (S D) T')
        return speca