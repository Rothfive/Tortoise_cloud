## Pitfalls You May Encounter

I'll try and make a list of "common" (or what I feel may be common that I experience) issues with getting TorToiSe set up:

### "I can't use the Web UI when it shows an error"

For god knows why, the Gradio web UI will prevent clicking on anything if it's listening outside of `127.0.0.1`.

You must click the (X) icon next to the error message to dismiss it before using the web UI again.

### `No hardware acceleration is available, falling back to CPU...`

Be sure you have used the right script for setup:
* NVIDIA: `setup-cuda.bat` / `setup-cuda.sh`
* AMD: `setup-directml.bat` / `setup-rocm.sh`

On NVIDIA, it seems some users require additional drivers for CUDA capabilities to be exposed. I'm not too sure about this myself, as I have the bare minimum drivers for my 2060, and might have gotten some CUDA runtimes through Nsight.

On Linux + AMD, you also need to ensure you have the ROCm-capable drivers/runtime installed. Please consult your distro's literature on how to install ROCm-capable drivers/runtime.

On Windows + AMD, I'm not too sure how this would be thrown, as DirectML does some DX12 wizardry for compute.

### `failed reading zip archive: failed finding central directory`

You had a file fail to download completely during the model downloading initialization phase.

Please open either `.\models\tortoise\` or `.\models\transformers\`, and delete the offending file.

You can deduce what that file is by reading the stack trace. A few lines above the last like will be a line trying to read a model path.

### Voicefixer is taking forever to download

Lately, it seems it just takes way too long to download Voicefixer's models. Just be patient.

### `torch.cuda.OutOfMemoryError: CUDA out of memory.`

#### Generation

You most likely have a GPU with low VRAM (~4GiB), and the small optimizations with keeping data on the GPU is enough to OOM. Please check the `Low VRAM` option under the `Settings` tab.

If you do have a beefy GPU:
* if you have very large voice input files, increase the `Voice Chunk` slider, as the scripts will try and compute a voice's latents in pieces, rather than in one chunk.
* if you're trying to generate a long sentence, please break your sentences into pieces, and set the `Line Delimiter` to `\n`.
* if you're simply trying to generate something small, please reduce your `Sample Batch Size` under the `Settings` tab.
* if you're getting this during a `voicefixer` pass, while using CUDA for it is enabled, please try disabling CUDA for Voice Fixer under the `Settings` tab, as it has its own model it loads into VRAM.
* if you're trying to create an LJSpeech dataset under `Train` > `Prepare Dataset`, please use a smaller Whisper model size under `Settings`.

#### Training

On Pascal-and-before cards, training is pretty much an impossible feat, as consumer cards lack the VRAM necessary to train, or the dedicated silicon to leverage optimizations like BitsAndBytes.

If you have a Turing (or beyond) card, you may have too large of a batch size, or a mega batch factor. Please try and reduce it before trying again, and ensure TorToiSe is NOT loaded by using the `Do Not Load TTS On Startup` option and restarting the web UI.

If you're in dire need to train, please try to train on a Colab notebook.

### `WavFileWarning: Chunk (non-data) not understood, skipping it.`

This is a rather innocuous error. I don't think generation quality is impacted at all, but if you insist on making it go away, remux your WAVs with something like `ffmpeg`.

### `AttributeError: module 'ffmpeg' has no attribute 'input'`

The python package `ffmpeg-python` is very meticulous when installing it through openai/whisper. In a command prompt, with the current working directory set to the repo, run:
* Windows:
```
call .\venv\Scripts\activate.bat
python -m pip uninstall ffmpeg ffmpeg-python
python -m pip install ffmpeg-python
```
* Linux:
```
source ./venv/bin/activate
pip uninstall ffmpeg ffmpeg-python
pip install torch ffmpeg-python
```

### `AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'`

You more than likely need to reinstall PyTorch. In a command prompt, with the current working directory set to the repo, run:
* Windows:
```
call .\venv\Scripts\activate.bat
python -m pip uninstall torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
* Linux:
```
source ./venv/bin/activate
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### `[WinError 5] Access is denied:`

This occurs when you:
* finetune from zero
* stop a training process
* start finetuning again from zero
* DLAS will attempt to backup and move the old folder, but because of a zombie Python process still having "ownership" of the folder, the folder cannot be removed

Open a command prompt and type `tskill python` to kill all Python processes. Relaunch the web UI, and try to train again.

### `local_state[k] = v[grad_accum_step] / IndexError: list index out of range`

Your `Gradiant Accumulation Size` is too large for your given `Batch Size`. Please reduce it to, at most, half your batch size, or use the validation button to correct this.

## Reporting Other Errors

I do not have all possible errors documented, so if you encounter one that you can't resolve, please open an Issue with adequate information, including:
* python version
* GPU
* stack trace (or full console output preferred), wrapped in \`\`\`\[trace\]\`\`\` 
* summary of what you were doing

and I'll try my best to remedy it, even if it's something small like not reading the documentation.

***Please, please, please*** provide either a full stack trace of the error (if running the web UI) or the command prompt output (if running a script). I will not know what's wrong if you only provide the error message itself, as errors are heavily predicated on the full state it happened. Without it, I cannot help you, as I would only be able to make assumptions.
