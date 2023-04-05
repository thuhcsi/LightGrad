""" from https://github.com/keithito/tacotron """

import re
from text import en_cleaners
from g2p_en import G2p

valid_symbols = [
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1',
    'AH2', 'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0',
    'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0',
    'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0',
    'IH1', 'IH2', 'IY', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
    'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W',
    'Y', 'Z', 'ZH'
]

_punctuation = '!\'(),.:;? '
_special = ['-', '<blank>', '<bos>', '<eos>']
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_alt_re = re.compile(r'\([0-9]+\)')

_arpabet = [s for s in valid_symbols]

# Export all symbols:
symbols = _special + list(_punctuation) + _arpabet
_valid_symbol_set = set(valid_symbols)

# zero is reserved for padding
_symbol_to_id = {s: i + 1 for i, s in enumerate(symbols)}
_id_to_symbol = {i + 1: s for i, s in enumerate(symbols)}

_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(en_cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
        text = re.sub('-','',text)
    return text


class G2pEn():

    def __init__(self) -> None:
        self.g2p = G2p()

    def __call__(self, text):
        phonemes = self.g2p(_clean_text(text, ["english_cleaners"]))
        text = ' '.join(phonemes)
        text = re.sub('  ,',',',text)
        text = re.sub(',  ',',',text)
        text = re.sub('  !','!',text)
        text = re.sub('!  ','!',text)
        text = re.sub('\?  ','?',text)
        text = re.sub('  \?','?',text)
        text = re.sub("  '","'",text)
        text = re.sub("'  ","'",text)
        text = re.sub("   "," - ",text)
        return text.split(' ')

