import argparse
import pathlib
import random
import re
import json
import tqdm


# for BZNSYP, 200 samples for test, 200 samples for validation
# for LJSpeech, 523 samples for test, 348 samples for validation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["ljspeech", "bznsyp"])
    parser.add_argument("dataset_path", type=str, help="path to dataset dir")
    parser.add_argument("export_dir", type=str, help="path to save preprocess result")
    parser.add_argument("--test_sample_count", type=int)
    parser.add_argument("--valid_sample_count", type=int)
    return parser.parse_args()


def main():
    args = get_args()
    if args.dataset == "ljspeech":
        (train_dataset, valid_dataset, test_dataset, phn2id) = preprocess_ljspeech(args)
    if args.dataset == "bznsyp":
        (train_dataset, valid_dataset, test_dataset, phn2id) = preprocess_bznsyp(args)
    export_dir = pathlib.Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    with open(export_dir / "train_dataset.json", "w") as f:
        json.dump(train_dataset, f)
    with open(export_dir / "valid_dataset.json", "w") as f:
        json.dump(valid_dataset, f)
    with open(export_dir / "test_dataset.json", "w") as f:
        json.dump(test_dataset, f)
    with open(export_dir / "phn2id.json", "w") as f:
        json.dump(phn2id, f)


def preprocess_ljspeech(args):
    from text import G2pEn, phn2id_en

    dataset_path = pathlib.Path(args.dataset_path)
    metadata_path = dataset_path / "metadata.csv.txt"
    meta_info = []
    g2p = G2pEn()
    with open(metadata_path) as f:
        for line in tqdm.tqdm(f.readlines()):
            name, _, normalized_text = line.strip().split("|")
            wav_path = dataset_path / "wavs" / f"{name}.wav"
            if wav_path.exists():
                phonemes = g2p(normalized_text)
                meta_info.append(
                    {
                        "name": name,
                        "wav_path": str(wav_path),
                        "text": normalized_text,
                        "phonemes": phonemes,
                    }
                )
    random.shuffle(meta_info)
    test_dataset = meta_info[: args.test_sample_count]
    valid_dataset = meta_info[
        args.test_sample_count : args.test_sample_count + args.valid_sample_count
    ]
    train_dataset = meta_info[args.test_sample_count + args.valid_sample_count :]
    return train_dataset, valid_dataset, test_dataset, phn2id_en


def preprocess_bznsyp(args):
    from text import G2pZh

    dataset_path = pathlib.Path(args.dataset_path)
    metadata_path = dataset_path / "metadata.csv.txt"
    meta_info = []
    g2p = G2pZh()
    with open(metadata_path) as f:
        for line in tqdm.tqdm(f.readlines()):
            name, text, pinyin = line.strip().split("|")
            # bznsyp using ng1 as en1
            pinyin = re.sub("ng1-yuan4-le5", "en1-yuan4-le5", pinyin)
            # correct english character
            pinyin = re.sub("P-IY1-guo4", "pi1-guo4", pinyin)
            pinyin = re.sub("/", " / ", pinyin)
            pinyin = re.sub("\?", " ? ", pinyin)
            pinyin = re.sub("!", " ! ", pinyin)
            pinyin = re.sub(",", " , ", pinyin)
            pinyin = re.sub(" {2,}", " ", pinyin)
            pinyin = re.sub(" $", "", pinyin)
            pinyin = re.sub("-", " ", pinyin)
            pinyin = re.sub("\.", "", pinyin)
            wav_path = dataset_path / "Wave.48k" / f"{name}.wav"
            if wav_path.exists():
                phonemes = g2p.pinyin2phoneme(pinyin)
                meta_info.append(
                    {
                        "name": name,
                        "wav_path": str(wav_path),
                        "text": text,
                        "phonemes": phonemes,
                    }
                )
    random.shuffle(meta_info)
    test_dataset = meta_info[: args.test_sample_count]
    valid_dataset = meta_info[
        args.test_sample_count : args.test_sample_count + args.valid_sample_count
    ]
    train_dataset = meta_info[args.test_sample_count + args.valid_sample_count :]
    return train_dataset, valid_dataset, test_dataset, g2p.phn2id()


if __name__ == "__main__":
    main()
