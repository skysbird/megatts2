'''
    wavs dir
    ├── speaker1
    │   ├── s1wav1.wav
    │   ├── s1wav1.txt
    │   ├── s1wav2.wav
    │   ├── s1wav2.txt
    │   ├── ...
    ├── speaker2
    │   ├── s2wav1.wav
    │   ├── s2wav1.txt
    │   ├── ...

    cautions: stage 0 will delete all txt files in wavs dir
'''

import os

import glob
from modules.tokenizer import TextTokenizer
from multiprocessing import Pool
from tqdm.auto import tqdm
from utils.textgrid import read_textgrid

import argparse

from lhotse import validate_recordings_and_supervisions, CutSet, NumpyHdf5Writer, load_manifest_lazy, load_manifest
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.recipes.utils import read_manifests_if_cached
from lhotse.utils import Seconds, compute_num_frames

from functools import partial

from modules.tokenizer import (
    HIFIGAN_SR,
    HIFIGAN_HOP_LENGTH,
    MelSpecExtractor,
    AudioFeatExtraConfig
)
# from models.megatts2 import MegaG
from models.gan2 import VQGANTTS

from modules.datamodule import TTSDataset, make_spk_cutset

from utils.symbol_table import SymbolTable

import soundfile as sf
import librosa

import torch
import numpy as np
import json
import codecs


def make_lab(tt, wav):
    id = wav.split('/')[-1].split('.')[0]
    folder = '/'.join(wav.split('/')[:-1])
    # Create lab files
    with open(f'{folder}/{id}.txt', 'r') as f:
        txt = f.read()

        with open(f'{folder}/{id}.lab', 'w') as f:
            f.write(' '.join(tt.tokenize(txt)))


from typing import Dict


punctuation = '!,.?"'


def get_phoneme_durations(
    data: Dict, original_text: str, fs: int, hop_size: int, n_samples: int
):
    """Get phoneme durations."""
    orig_text = original_text.replace(r"\s", " ").rstrip()
    # orig_text = original_text.rstrip()
    text_pos = 0
    maxTimestamp = data["end"]
    words = [x for x in data["tiers"]["words"]["entries"]]
    phones = (x for x in data["tiers"]["phones"]["entries"])
    word_end = 0
    time2punc = {}
    phrase_words = []
    last_punc = []

    for word in words:
        start, end, label = word
        if start == word_end:
            phrase_words.append(label)
            if last_punc:
                timing = (word_end, word_end)
                time2punc[timing] = last_punc

            # find punctuation at end of previous phrase
            phrase = label
            for letter in phrase:
                char = orig_text[text_pos]
                while char != letter:
                    text_pos += 1
                    char = orig_text[text_pos]
                text_pos += 1

            last_punc = []
            while text_pos < len(orig_text):
                char = orig_text[text_pos]
                if char.isalpha() or char == "'":
                    break
                else:
                    last_punc.append(char)
                    # puncs.append(" ")
                    text_pos += 1

            # time2punc[timing] = puncs if puncs else [" "]

        else:
            # find punctuation at end of previous phrase
            # phrase = "".join(phrase_words)
            if last_punc:
                timing = (word_end, start)
                time2punc[timing] = last_punc
            
            phrase = label
            for letter in phrase:
                char = orig_text[text_pos]
                while char != letter:
                    text_pos += 1
                    char = orig_text[text_pos]
                text_pos += 1

            last_punc = []
            while text_pos < len(orig_text):
                char = orig_text[text_pos]
                if char.isalpha() or char == "'":
                    break
                else:
                    last_punc.append(char)
                    # puncs.append(" ")
                    text_pos += 1
            # print("a",timing)


            phrase_words = [label]
        
        #add space
        # print("add space")
        # print(phrase_words)
        # time2punc[(word_end,word_end)] = [' ']
        word_end = end

    print(time2punc)
    # We preserve the start/end timings and not interval lengths
    #   due to rounding errors when converted to frames.
    timings = [0.0]
    new_phones = []
    prev_end = 0.0
    for phone in phones:
        start, end, label = phone
        if start == prev_end:
            try:
                puncs = time2punc[(prev_end, start)]
                new_phones.extend(puncs)
                num_puncs = len(puncs)
                pause_time = (start - prev_end) / num_puncs
                for i in range(1, len(puncs) + 1):
                    timings.append(prev_end + pause_time * i)
            except KeyError:
                print("add space wrong")
                pass

        if start > prev_end:
            # insert punctuation
            try:
                puncs = time2punc[(prev_end, start)]
            except KeyError:
                # In some cases MFA word segmentation fails
                #   and there is a pause inside the word.
                puncs = ["sil"]
            new_phones.extend(puncs)
            num_puncs = len(puncs)
            pause_time = (start - prev_end) / num_puncs
            for i in range(1, len(puncs) + 1):
                timings.append(prev_end + pause_time * i)

        new_phones.append(label)
        timings.append(end)
        prev_end = end

    # Add end-of-utterance punctuations
    if word_end < maxTimestamp:
        text_pos = len(orig_text) - 1
        while orig_text[text_pos] in punctuation:
            text_pos -= 1
        puncs = orig_text[text_pos + 1 :]
        if not puncs:
            puncs = ["sil"]
        new_phones.extend(puncs)
        num_puncs = len(puncs)
        pause_time = (maxTimestamp - word_end) / num_puncs
        for i in range(1, len(puncs) + 1):
            timings.append(prev_end + pause_time * i)

    assert len(new_phones) + 1 == len(timings)

    # Should use same frame formulation for both
    # STFT frames calculation: https://github.com/librosa/librosa/issues/1288
    # centered stft

    total_durations = int(n_samples / hop_size) + 1
    timing_frames = [int(timing * fs / hop_size) + 1 for timing in timings]
    durations = [
        timing_frames[i + 1] - timing_frames[i] for i in range(len(timing_frames) - 1)
    ]

    sum_durations = sum(durations)
    if sum_durations < total_durations:
        missing = total_durations - sum_durations
        durations[-1] += missing
    assert sum(durations) == total_durations
    assert len(new_phones) == len(durations)
    return new_phones[:], durations[:]

def make_dp(tgt_path, wav_path, text):
    with codecs.open(tgt_path, "r", encoding="utf-8") as reader:
        _data_dict = json.load(reader)

    with sf.SoundFile(wav_path) as audio:
        orig_sr = audio.samplerate
        # Account for downsampling
        no_samples = int(audio.frames * (24_000 / orig_sr))

    hop_size = 256
    sr = 24_000
    p,d = get_phoneme_durations(_data_dict, text.lower(), sr, hop_size, no_samples)
    return p,d


class DatasetMaker:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--stage', type=int, default=0,
                            help='Stage to start from')
        parser.add_argument('--wavtxt_path', type=str,
                            default='data/wavs/', help='Path to wav and txt files')
        parser.add_argument('--text_grid_path', type=str,
                            default='data/textgrids/', help='Path to textgrid files')
        parser.add_argument('--ds_path', type=str,
                            default='data/ds/', help='Path to save dataset')
        parser.add_argument('--num_workers', type=int,
                            default=4, help='Number of workers')
        parser.add_argument('--test_set_ratio', type=float,
                            default=0.03, help='Test set ratio')
        parser.add_argument('--trim_wav', type=bool,
                            default=False, help='Trim wav by textgrid')
        parser.add_argument('--generator_ckpt', type=str,
                            default='generator.ckpt', help='Load generator checkpoint')
        parser.add_argument('--generator_config', type=str,
                            default='configs/config_gan.yaml', help='Load generator config')

        self.args = parser.parse_args()

        self.test_set_interval = int(1 / self.args.test_set_ratio)

    def make_labs(self):
        wavs = glob.glob(f'{self.args.wavtxt_path}/**/*.wav', recursive=True)
        tt = TextTokenizer()

        with Pool(self.args.num_workers) as p:
            for _ in tqdm(p.imap(partial(make_lab, tt), wavs), total=len(wavs)):
                pass

    def make_ds(self):
        tgs = glob.glob(
            f'{self.args.text_grid_path}/**/*.TextGrid', recursive=True)

        recordings = [[], []]  # train, test
        supervisions = [[], []]
        set_name = ['train', 'valid']
        max_duration_token = 0

        for i, tg in tqdm(enumerate(tgs)):
            id = tg.split('/')[-1].split('.')[0]
            speaker = tg.split('/')[-2]

            # intervals = [i for i in read_textgrid(tg) if (i[3] == 'phones')]

            # y, sr = librosa.load(
            #     f'{self.args.wavtxt_path}/{speaker}/{id}.wav', sr=HIFIGAN_SR)

            # if intervals[0][2] == '':
            #     intervals = intervals[1:]
            # if intervals[-1][2] == '':
            #     intervals = intervals[:-1]
            # if self.args.trim_wav:
            #    start = intervals[0][0]*sr
            #    stop = intervals[-1][1]*sr
            #    y = y[int(start):int(stop)]
            #    y = librosa.util.normalize(y)

            #    sf.write(
            #        f'{self.args.wavtxt_path}/{speaker}/{id}.wav', y, HIFIGAN_SR)

            # start = intervals[0][0]
            # stop = intervals[-1][1]

            # frame_shift=HIFIGAN_HOP_LENGTH / HIFIGAN_SR
            # duration = round(y.shape[-1] / HIFIGAN_SR, ndigits=12)
            # n_frames = compute_num_frames(
            #     duration=duration,
            #     frame_shift=frame_shift,
            #     sampling_rate=HIFIGAN_SR,
            # )

            # duration_tokens = []
            # phone_tokens = []

            # for i, interval in enumerate(intervals):

            #     phone_stop = (interval[1] - start)
            #     n_frame_interval = int(phone_stop / frame_shift)
            #     duration_tokens.append(n_frame_interval - sum(duration_tokens))
            #     phone_tokens.append(interval[2] if interval[2] != '' else '<sil>')

            # if sum(duration_tokens) > n_frames:
            #     print(f'{id} duration_tokens: {sum(duration_tokens)} must <= n_frames: {n_frames}')
            #     assert False

            recording = Recording.from_file(
                f'{self.args.wavtxt_path}/{speaker}/{id}.wav')
            text = open(
                f'{self.args.wavtxt_path}/{speaker}/{id}.txt', 'r').read()
            
            phone_tokens, duration_tokens = make_dp(tg, f'{self.args.wavtxt_path}/{speaker}/{id}.wav', text) 


            segment = SupervisionSegment(
                id=id,
                recording_id=id,
                start=0,
                duration=recording.duration,
                channel=0,
                language="EN",
                speaker=speaker,
                text=text,
            )

            # if abs(recording.duration - (stop - start)) > 0.01:
            #     print(f'{id} recording duration: {recording.duration} != {stop - start}')
            #     assert False

            set_id = 0 if i % self.test_set_interval else 1
            recordings[set_id].append(recording)
            supervisions[set_id].append(segment)

            segment.custom = {}
            segment.custom['duration_tokens'] = duration_tokens
            segment.custom['phone_tokens'] = phone_tokens

            max_duration_token = max(max_duration_token, len(duration_tokens))

            assert len(duration_tokens) == len(phone_tokens)

        print(recordings)
        print(supervisions)

        for i in range(2):
            recording_set = RecordingSet.from_recordings(recordings[i])
            supervision_set = SupervisionSet.from_segments(supervisions[i])
            validate_recordings_and_supervisions(
                recording_set, supervision_set)

            supervision_set.to_file(
                f"{self.args.ds_path}/supervisions_{set_name[i]}.jsonl.gz")
            recording_set.to_file(
                f"{self.args.ds_path}/recordings_{set_name[i]}.jsonl.gz")

        # Extract features
        manifests = read_manifests_if_cached(
            dataset_parts=['train', 'valid'],
            output_dir=self.args.ds_path,
            prefix="",
            suffix='jsonl.gz',
            types=["recordings", "supervisions"],
        )

        for partition, m in manifests.items():
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )

            # extract
            cut_set = cut_set.compute_and_store_features(
                extractor=MelSpecExtractor(AudioFeatExtraConfig()),
                storage_path=f"{self.args.ds_path}/cuts_{partition}",
                storage_type=NumpyHdf5Writer,
                num_jobs=self.args.num_workers,
            )

            cut_set.to_file(
                f"{self.args.ds_path}/cuts_{partition}.jsonl.gz")
            
        print(f'max_duration_token: {max_duration_token}')

    def extract_latent(self):


        def filter_duration(c):
            if c.duration < 0 or c.duration > 20:
                return False
            return True

        os.system(f'mkdir -p {self.args.ds_path}/latents')

        G = VQGANTTS.from_pretrained(dm.args.generator_ckpt, dm.args.generator_config)
        #G = MegaG.from_pretrained(dm.args.generator_ckpt, dm.args.generator_config)
        #G = VQGANTTS.from_hparams(dm.args.generator_config)
        G = G.cuda()
        G.eval()

        cs_all = load_manifest(f'{dm.args.ds_path}/cuts_train.jsonl.gz') + load_manifest(f'{dm.args.ds_path}/cuts_valid.jsonl.gz')
        spk_cs = make_spk_cutset(cs_all)

        for spk in spk_cs.keys():
            os.system(f'mkdir -p {self.args.ds_path}/latents/{spk}')

        ttsds = TTSDataset(spk_cs, f'{dm.args.ds_path}', 10)

        for c in tqdm(cs_all):
            if filter_duration(c) == False:
                print("duration skip",c.duration)
                continue
            id = c.recording_id
            spk = c.supervisions[0].speaker
            batch = ttsds.__getitem__(CutSet.from_cuts([c]))

            s2_latent = {}
            with torch.no_grad():

                tc_latent, p_code = G.s2_latent(
                    batch['phone_tokens'].cuda(),
                    batch['src_pos'].cuda(),
                    batch['mel_targets'].cuda(),
                    batch['mel_timbres'].cuda(),
                    batch['duration_tokens'].cuda(),
                )

                #print(batch)
                print("tttt")
                print(tc_latent.shape)
                print(batch['duration_tokens'].shape)
                print("uuuuuu")

                s2_latent['tc_latent'] = tc_latent.cpu().numpy()
                s2_latent['p_code'] = p_code.cpu().numpy()

            np.save(f'{self.args.ds_path}/latents/{spk}/{id}.npy', s2_latent)

if __name__ == '__main__':
    dm = DatasetMaker()

    # Create lab files
    if dm.args.stage == 0:
        dm.make_labs()
    elif dm.args.stage == 1:
        dm.make_ds()

        # Test
        cs_train = load_manifest_lazy(
            f'{dm.args.ds_path}/cuts_train.jsonl.gz')
        cs_valid = load_manifest_lazy(
            f'{dm.args.ds_path}/cuts_valid.jsonl.gz')
        cs = cs_train + cs_valid

        unique_symbols = set()

        for c in tqdm(cs):
            unique_symbols.update(c.supervisions[0].custom["phone_tokens"])

        unique_phonemes = SymbolTable()
        for s in sorted(list(unique_symbols)):
            unique_phonemes.add(s)

        unique_phonemes_file = f"unique_text_tokens.k2symbols"
        unique_phonemes.to_file(f'{dm.args.ds_path}/{unique_phonemes_file}')

        print(cs.describe())
    elif dm.args.stage == 2:
        dm.extract_latent()




# if __name__== "__main__":
#     tgt_path = '100_121669_000001_000000.json'

#     wav_path = '100_121669_000001_000000.wav'


#     p,d = make_dp(tgt_path, wav_path, "Tom, the Piper's Son".lower())
#     print(p)
#     print(d)