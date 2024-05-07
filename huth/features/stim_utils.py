import numpy as np
import huth.config as config
from huth.data.textgrid import TextGrid
from huth.data.data_sequence import DataSequence
from huth.data.utils_ds import make_word_ds
import json
from typing import Dict, List
import os
from os.path import join, dirname


def load_story_wordseqs(stories) -> Dict[str, DataSequence]:
    # load textgrids
    base = join(config.root_dir, "ds003020/derivative/TextGrids")
    grids = {}
    for story in stories:
        grid_path = os.path.join(base, f"{story}.TextGrid")
        grids[story] = TextGrid(open(grid_path).read())

    # make into wordseqs
    with open(join(config.root_dir, "ds003020/derivative/respdict.json"), "r") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    wordseqs = make_word_ds(grids, trfiles)
    return wordseqs


class TRFile(object):
    def __init__(self, trfilename, expectedtr=2.0045):
        """Loads data from [trfilename], should be output from stimulus presentation code.
        """
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr

        if trfilename is not None:
            self.load_from_file(trfilename)

    def load_from_file(self, trfilename):
        """Loads TR data from report with given [trfilename].
        """
        # Read the report file and populate the datastructure
        for ll in open(trfilename):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)

            if label in ("init-trigger", "trigger"):
                self.trtimes.append(time)

            elif label == "sound-start":
                self.soundstarttime = time

            elif label == "sound-stop":
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))

        # Fix weird TR times
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes > (itrtimes.mean()*1.5))[0]
        newtrs = []
        for btr in badtrtimes:
            # Insert new TR where it was missing..
            newtrtime = self.trtimes[btr]+self.expectedtr
            newtrs.append((newtrtime, btr))

        for ntr, btr in newtrs:
            self.trtimes.insert(btr+1, ntr)

    def simulate(self, ntrs):
        """Simulates [ntrs] TRs that occur at the expected TR.
        """
        self.trtimes = list(np.arange(ntrs)*self.expectedtr)

    def get_reltriggertimes(self):
        """Returns the times of all trigger events relative to the sound.
        """
        return np.array(self.trtimes)-self.soundstarttime

    @property
    def avgtr(self):
        """Returns the average TR for this run.
        """
        return np.diff(self.trtimes).mean()


def load_simulated_trfiles(respdict, tr=2.0, start_time=10.0, pad=5):
    trdict = dict()
    for story, resps in respdict.items():
        trf = TRFile(None, tr)
        trf.soundstarttime = start_time
        trf.simulate(resps - pad)
        trdict[story] = [trf]
    return trdict
