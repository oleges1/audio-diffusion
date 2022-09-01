import torch
from torch.utils import data
import torchaudio
import random
from torchaudio import datasets
import torchaudio.functional as AF

from einops import rearrange

hann_window = {}

def spectrogram(y, n_fft, hop_size, win_size, center=False):
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

class DRVCTK(datasets.DR_VCTK):
    def __init__(
            self, root, segment_size, n_fft, 
            hop_size, win_size, raw_wave,
            subset='train', zero_out_percent=None, transform=None
        ):
        super().__init__(root, subset, download=False)
        self.subset = subset
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.zero_out_percent = zero_out_percent
        self.raw_wave = raw_wave
        self.transform = transform

    
    def __getitem__(self, i):
        file_clean_audio = self._clean_audio_dir / self._filename_list[i]
        audio, sample_rate_clean = torchaudio.load(file_clean_audio)

        audio = torch.FloatTensor(audio)
        # audio = audio.unsqueeze(0)

        if audio.size(1) >= self.segment_size:
            max_audio_start = audio.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start+self.segment_size]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
        
        if self.raw_wave:
            return audio, {}

        speca = spectrogram(audio, self.n_fft, self.hop_size, self.win_size,
                                   center=False)
        
        if self.zero_out_percent is not None:
            speca[:, speca.shape[1] // 2:] = 0
        
        speca = rearrange(speca, 'B S T D -> B (S D) T')
        # return speca.squeeze(0), {}
        return self.transform(speca), {}
    
    

class VCTK(datasets.VCTK_092):
    """
    VCTK.092
    """

    def __init__(
            self, root, segment_size, n_fft, 
            hop_size, win_size, raw_wave,
             zero_out_percent=None, transform=None
        ):
        super().__init__(root, download=False)
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.zero_out_percent = zero_out_percent
        self.raw_wave = raw_wave
        self.transform = transform

    def __getitem__(self, i):
        
        speaker_id, utterance_id = self._sample_ids[i]
        audio, sample_rate, transcript, speaker_id, utterance_id = self._load_sample(speaker_id, utterance_id, self._mic_id)

        audio = torch.FloatTensor(audio)
        # audio = audio.unsqueeze(0)

        if audio.size(1) >= self.segment_size:
            max_audio_start = audio.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start+self.segment_size]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        if self.raw_wave:
            return audio, {}

        speca = spectrogram(audio, self.n_fft, self.hop_size, self.win_size,
                                   center=False)

        if self.zero_out_percent is not None:
            speca[:, speca.shape[1] // 2:] = 0

        speca = rearrange(speca, 'B S T D -> B (S D) T')

        return self.transform(speca), {}
