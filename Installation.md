This software has been tested both locally on Windows and Linux, as well as Jupyter instances.

## Installing

Outside of the very small prerequisites, everything needed to get TorToiSe working is included in the repo.

### Pre-Requirements

Windows:
* Python: https://www.python.org/downloads/windows/
	- Tested on python3.9: https://www.python.org/downloads/release/python-3913/
    - Briefly tested on python3.10
* Git: https://git-scm.com/download/win
* CUDA drivers, if NVIDIA
* FFMPEG: https://ffmpeg.org/download.html#build-windows
	- only needed when preparing datasets for training/finetuning

Linux:
* python3.x (tested with 3.10)
* git
* ROCm for AMD, CUDA for NVIDIA
* FFMPEG:
	- only needed when preparing datasets for training/finetuning

#### CUDA Version

For NVIDIA cards, the setup script assumes your card support CUDA 11.7. If your GPU does not, simply edit the setup script to the right CUDA version. For example: `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113` instead.

### Setup

#### Windows

Download Python and Git and run their installers.

After installing Python, open the Start Menu and search for `Command Prompt`. Type `cd `, then drag and drop the folder you want to work in (experienced users can just `cd <path>` directly), then hit Enter.

Paste `git clone https://git.ecker.tech/mrq/ai-voice-cloning` to download TorToiSe and additional scripts, then hit Enter.

Afterwards, run the setup script, depending on your GPU, to automatically set things up.
* AMD: `setup-directml.bat`
* NVIDIA: `setup-cuda.bat`

If you've done everything right, you shouldn't have any errors.

##### Note on DirectML Support

PyTorch-DirectML is very, very experimental and is still not production quality. There's some headaches with the need for hairy kludgy patches.

These patches rely on transfering the tensor between the GPU and CPU as a hotfix, so performance is definitely harmed.

Both the conditional latent computation and the vocoder pass have to be done on the CPU entirely because of some quirks with DirectML.

On my 6800XT, VRAM usage climbs almost the entire 16GiB, so be wary if you OOM somehow. Low VRAM flags may NOT have any additional impact from the constant copying anyways.

For AMD users, I still might suggest using Linux+ROCm as it's (relatively) headache free, but I had stability problems.

Training is currently very, very improbably, due to how integrated it seems to be with CUDA. If you're fiending to train on your AMD card, please use Linux+ROCm, but I have not tested this myself.

#### Linux

First, make sure you have both `python3.x` and `git` installed, as well as the required compute platform according to your GPU (ROCm or CUDA).

Simply run the following block:

```
git clone https://git.ecker.tech/mrq/ai-voice-cloning
cd tortoise-tts
chmod +x *.sh
```

Then, depending on your GPU:
* AMD: `./setup-rocm.sh`
* NVIDIA: `./setup-cuda.sh`

And you should be done!

#### Note for AMD users

Due to the nature of ROCm, some little problems may occur.

Additionally, training on AMD cards cannot leverage BitsAndBytes optimizations, as those are tied to CUDA runtimes.

### Updating

To check for updates, simply run `update.bat` (or `update.sh`). It should pull from the repo, as well as fetch for any new dependencies.

## Notebooks

The repo provides a notebooks to run the software on an instance rather than local hardware.

### Colab
[Notebook](https://git.ecker.tech/mrq/ai-voice-cloning/raw/branch/master/notebook_colab.ipynb): https://git.ecker.tech/mrq/ai-voice-cloning/raw/branch/master/notebook_colab.ipynb

For colab use, simply:
* download the above notebook
* navigate to [Colab](https://colab.research.google.com/)
* upload the notebook
* run the Installation block
* run the Drive Mount block
* run EITHER the "Run (Inlined)" or "Run (Non-inlined)"
* add in your voices through Google Drive (if mounted) or upload them on the sidebar on the left

### Paperspace
[Notebook](https://git.ecker.tech/mrq/ai-voice-cloning/raw/branch/master/notebook_paperspace.ipynb): https://git.ecker.tech/mrq/ai-voice-cloning/raw/branch/master/notebook_paperspace.ipynb

For paperspace use, simply:
* download the above notebook
* navigate to [Paperspace](https://www.paperspace.com/)
* create either a new account or login to an existing one
* create a new Gradient project
* create a notebook
* click start from scratch
* select your machine (this can be changed later on)
* select `6 hours` for auto-shutdown time (this can be changed later)
* click start notebook
* upload the notebook file
* run the Installation block
* run the Running block