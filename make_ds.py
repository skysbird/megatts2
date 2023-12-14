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

from lhotse import validate_recordings_and_supervisions, CutSet, NumpyHdf5Writer
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.recipes.utils import read_manifests_if_cached

from functools import partial

from modules.tokenizer import MelSpecExtractor, AudioFeatExtraConfig

DURATION_MAX = 100  # 10ms ~ 1s


def make_lab(tt, wav):
    id = wav.split('/')[-1].split('.')[0]
    folder = '/'.join(wav.split('/')[:-1])
    # Create lab files
    with open(f'{folder}/{id}.txt', 'r') as f:
        txt = f.read()

        with open(f'{folder}/{id}.lab', 'w') as f:
            f.write(' '.join(tt.tokenize(txt)))


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
                            default=0.01, help='Test set ratio')

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

        for i, tg in tqdm(enumerate(tgs)):
            id = tg.split('/')[-1].split('.')[0]
            speaker = tg.split('/')[-2]

            intervals = [i for i in read_textgrid(tg) if i[3] == 'phones']

            duration_tokens = []
            phone_tokens = []
            for interval in intervals:
                if interval[2] == '':
                    continue

                duration = (interval[1] - interval[0]) * 100
                if duration > DURATION_MAX:
                    duration = DURATION_MAX

                duration_tokens.append(int(duration))
                phone_tokens.append(interval[2])

            recording = Recording.from_file(
                f'{self.args.wavtxt_path}/{speaker}/{id}.wav')
            text = open(
                f'{self.args.wavtxt_path}/{speaker}/{id}.txt', 'r').read()
            segment = SupervisionSegment(
                id=id,
                recording_id=id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language="CN",
                speaker=speaker,
                text=text,
            )
            set_id = 0 if i % self.test_set_interval else 1
            recordings[set_id].append(recording)
            supervisions[set_id].append(segment)

            segment.custom = {}
            segment.custom['duration_tokens'] = duration_tokens
            segment.custom['phone_tokens'] = phone_tokens

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
                f"{self.args.ds_path}/cuts_{partition}.json.gz")

if __name__ == '__main__':
    dm = DatasetMaker()

    # Create lab files
    if dm.args.stage == 0:
        dm.make_labs()
    elif dm.args.stage == 1:
        dm.make_ds()
        # with Pool(args.num_workers) as p:
        #     for _ in tqdm(p.imap(make_ds, tgs), total=len(tgs)):
        #         pass
