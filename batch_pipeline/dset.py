import os
import joblib
from torch.utils.data import Dataset
import h5py
import numpy as np
from huth.utils_ds import make_word_ds


class HuthLabDataset(Dataset):
    def __init__(self, data_dir: str, subject: str, trim_start: int = 5, trim_end: int = 10):
        self.subject = subject
        self.stories = joblib.load(os.path.join(
            data_dir, subject, "storylist.jbl"))
        self.grids = joblib.load(os.path.join(
            data_dir, "story_data", "grids.jbl"))
        self.trfiles = joblib.load(os.path.join(
            data_dir, "story_data", "trfiles.jbl"))
        self.wordseqs = make_word_ds(self.grids, self.trfiles)
        self.trim_start = trim_start
        self.trim_end = trim_end
        # self.lookback = num_trs  # num_trs = num_seconds / 2
        self.resp_dict = {}
        self.chunk_dict = {}
        for story in self.stories:
            hf5_path = os.path.join(data_dir, subject, story + ".hf5")
            self.resp_dict[story] = h5py.File(hf5_path, 'r')
            self.chunk_dict[story] = self.wordseqs[story].chunks()[
                self.trim_start:-self.trim_end]
            # Confirm trimming dimensions match
            num_trs_stim = len(
                self.wordseqs[story].tr_times[self.trim_start:-self.trim_end])
            num_trs_resp = self.resp_dict[story]['data'].shape[0]
            assert num_trs_stim == num_trs_resp

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, story: str, idx: int, delays: int):
        assert delays >= 0
        if delays == 0:
            return (self.chunk_dict[story][idx], self.resp_dict[story]['data'][idx])
        else:
            acc_out = []
            for i in range(delays+1):
                if idx-delays+i < 0:
                    acc_out.append(np.array([], dtype='<U13'))
                else:
                    acc_out.append(self.chunk_dict[story][idx-delays+i])
            return (acc_out, self.resp_dict[story]['data'][idx])


if __name__ == '__main__':
    dset = HuthLabDataset("data", "UTS01")
