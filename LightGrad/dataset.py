import torch
import torch.utils.data
import json
import math
from librosa.filters import mel as librosa_mel_fn
import re
import torchaudio

from torch.nn.utils.rnn import pad_sequence


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 datalist_path,
                 phn2id_path,
                 sample_rate,
                 n_fft,
                 n_mels,
                 fmin,
                 fmax,
                 hop_size,
                 win_size,
                 add_blank=True):
        super().__init__()
        with open(datalist_path) as f:
            self.datalist = json.load(f)
        with open(phn2id_path) as f:
            self.phone_set = json.load(f)

        self.add_blank = add_blank
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.hop_size = hop_size
        self.win_size = win_size
        self.cache = {}
        self.hann_window = torch.hann_window(win_size)
        self.mel_basis = torch.from_numpy(
            librosa_mel_fn(sr=sample_rate,
                           n_fft=n_fft,
                           n_mels=n_mels,
                           fmin=fmin,
                           fmax=fmax)).float()

    def get_vocab_size(self):
        # PAD is also considered
        return len(self.phone_set) + 1

    def load_audio_and_melspectrogram(self, audio_path):
        audio, original_sr = torchaudio.load(audio_path)
        if original_sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, original_sr,
                                                   self.sample_rate)
        audio = torch.nn.functional.pad(audio.unsqueeze(1), (int(
            (self.n_fft - self.hop_size) /
            2), int((self.n_fft - self.hop_size) / 2)),
                                        mode='reflect')
        audio = audio.squeeze(1)
        spec = torch.stft(audio,
                          self.n_fft,
                          self.hop_size,
                          self.win_size,
                          self.hann_window,
                          False,
                          onesided=True,
                          return_complex=True)
        spec = spec.abs()
        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec).squeeze(0)
        # audio: (1,T) spec: (T,n_mels)
        return audio, spec.T

    def load_item(self, i):
        #item_name, wav_path, text, phonemes = self.datalist[i]
        item_name = self.datalist[i]['name']
        wav_path = self.datalist[i]['wav_path']
        text = self.datalist[i]['text']
        phonemes = self.datalist[i]['phonemes']

        audio, mel = self.load_audio_and_melspectrogram(wav_path)
        if self.add_blank:
            phonemes = " <blank> ".join(phonemes).split(' ')
        phonemes = ['<bos>'] + phonemes + ['<eos>']
        ph_idx = [self.phone_set[x] for x in phonemes if x in self.phone_set]
        self.cache[i] = {
            'item_name': item_name,
            'txt': text,
            'wav': audio,
            'ph': phonemes,
            'mel': mel,
            'ph_idx': ph_idx
        }
        return self.cache[i]

    def __getitem__(self, i):
        return self.cache.get(i, self.load_item(i))

    def process_item(self, item):
        ph = item['ph']
        # remove original | because this indicates word boundary
        ph = re.sub(r' \|', '', ph).split(' ')
        if self.add_blank:
            # add | as the phoneme boundary
            ph = ' | '.join(ph).split(' ')
        new_item = {
            'item_name': item['item_name'],
            'txt': item['txt'],
            'ph': ph,
            'mel': item['mel'],
            'ph_idx': [self.phone_set[x] for x in ph if x in self.phone_set],
            'wav': item['wav'],
        }
        return new_item

    def __len__(self):
        return len(self.datalist)


def collateFn(batch):
    phs_lengths, sorted_idx = torch.sort(torch.LongTensor(
        [len(x['ph_idx']) for x in batch]),
                                         descending=True)

    mel_lengths = torch.tensor([batch[i]['mel'].shape[0] for i in sorted_idx])
    padded_phs = pad_sequence(
        [torch.tensor(batch[i]['ph_idx']) for i in sorted_idx],
        batch_first=True)

    padded_mels = pad_sequence([batch[i]['mel'] for i in sorted_idx],
                               batch_first=True)
    batch_size, old_t, mel_d = padded_mels.shape
    txts = [batch[i]['txt'] for i in sorted_idx]
    wavs = [batch[i]['wav'] for i in sorted_idx]
    item_names = [batch[i]['item_name'] for i in sorted_idx]
    if old_t % 4 != 0:
        new_t = int(math.ceil(old_t / 4) * 4)
        temp = torch.zeros((batch_size, new_t, mel_d))
        temp[:, :old_t] = padded_mels
        padded_mels = temp
    return {
        'x': padded_phs,
        'x_lengths': phs_lengths,
        'y': padded_mels.permute(0, 2, 1),
        'y_lengths': mel_lengths,
        'txts': txts,
        'wavs': wavs,
        'names': item_names
    }


if __name__ == '__main__':
    import tqdm
    #dataset = Dataset('dataset/bznsyp_processed/train_dataset.json',
    #                         'dataset/bznsyp_processed/phn2id.json', 22050,
    #                         1024, 80, 0, 8000, 256, 1024)
    dataset = Dataset('dataset/ljspeech_processed/train_dataset.json',
                      'dataset/ljspeech_processed/phn2id.json', 22050, 1024,
                      80, 0, 8000, 256, 1024)
    #for i in tqdm.tqdm(range(len(dataset))):
    #    dataset[i]
    data = collateFn([dataset[i] for i in range(2)])
    print(data['x'])
    print(data['x_lengths'])
    print(data['y'].shape)
    print(data['y_lengths'])
    print(data['txts'])
    print(data['names'])