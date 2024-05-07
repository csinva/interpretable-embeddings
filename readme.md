# Dataset set up
- download data with `python experiments/00_load_dataset.py`
    - create a `data` dir under wherever you run it and will use [datalad](https://github.com/datalad/datalad) to download the preprocessed data as well as feature spaces needed for fitting [semantic encoding models](https://www.nature.com/articles/nature17637)
- set `ridge_utils.config.root_dir` to where you want to store the data
- to make flatmaps, need to set [pycortex filestore] to `{root_dir}/ds003020/derivative/pycortex-db/`
- to run eng1000, need to grab `em_data` directory from [here](https://github.com/HuthLab/deep-fMRI-dataset) and move its contents to `{root_dir}/em_data`
- loading responses
  - `ridge_utils.data.response_utils` function `load_response`
  - loads responses from at `{root_dir}/ds003020/derivative/preprocessed_data/{subject}`, hwere they are stored in an h5 file for each story, e.g. `wheretheressmoke.h5`
- loading stimulus
  - `ridge_utils.features.stim_utils` function `load_story_wordseqs`
  - loads textgrids from `{root_dir}/ds003020/derivative/TextGrids", where each story has a TextGrid file, e.g. `wheretheressmoke.TextGrid`
  - uses `{root_dir}/ds003020/derivative/respdict.json` to get the length of each story

# Code install
- from the repo directory, start with `pip install -e .` to locally install the `huth` package
- `python 01_fit_encoding.py --subject UTS03 --feature eng1000`
    - The other optional parameters that encoding.py takes such as sessions, ndelays, single_alpha allow the user to change the amount of data and regularization aspects of the linear regression used. 
    - This function will then save model performance metrics and model weights as numpy arrays. 