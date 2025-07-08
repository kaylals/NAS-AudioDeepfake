# split_data.py
import random
from pathlib import Path

def split_protocol_3way(protocol_file='../../data/test_sample/protocol.txt', train_ratio=0.7, dev_ratio=0.15):
    with open(protocol_file, 'r') as f:
        lines = f.readlines()

    bonafide_lines = [l for l in lines if 'bonafide' in l]
    spoof_lines = [l for l in lines if 'spoof' in l]

    random.shuffle(bonafide_lines)
    random.shuffle(spoof_lines)

    def split_lines(lines, ratio):
        a = int(len(lines) * ratio)
        b = int(len(lines) * (ratio + dev_ratio))
        return lines[:a], lines[a:b], lines[b:]

    train_bon, dev_bon, eval_bon = split_lines(bonafide_lines, train_ratio)
    train_spoof, dev_spoof, eval_spoof = split_lines(spoof_lines, train_ratio)

    train_lines = train_bon + train_spoof
    dev_lines = dev_bon + dev_spoof
    eval_lines = eval_bon + eval_spoof

    random.shuffle(train_lines)
    random.shuffle(dev_lines)
    random.shuffle(eval_lines)

    base_path = Path(protocol_file).parent
    (base_path / 'train.txt').write_text(''.join(train_lines))
    (base_path / 'dev.txt').write_text(''.join(dev_lines))
    (base_path / 'eval.txt').write_text(''.join(eval_lines))

if __name__ == "__main__":
    split_protocol_3way()
