## Settings

This tab (should) hold a bunch of other settings, from tunables that shouldn't be tampered with, to settings pertaining to the web UI itself.

Below are settings that override the default launch arguments. Some of these require restarting to work.
* `Listen`: sets the hostname, port, and/or path for the web UI to listen on.
	- For example, `0.0.0.0:80` will have the web UI accept all connections on port 80
	- For example, `10.0.0.1:8008/gradio` will have the web UI only accept connections through `10.0.0.1`, at the path `/gradio`
* `Public Share Gradio`: Tells Gradio to generate a public URL for the web UI. Ignored if specifying a path through the `Listen` setting.
* `Check for Updates`: checks for updates on page load and notifies in console. Only works if you pulled this repo from a gitea instance.
* `Only Load Models Locally`: enforces offline mode for loading models. This is the equivalent of setting the env var: `TRANSFORMERS_OFFLINE`
* `Low VRAM`: disables optimizations in TorToiSe that increases VRAM consumption. Suggested if your GPU has under 6GiB.
* `Embed Output Metadata`: enables embedding the settings and latents used to generate that audio clip inside that audio clip. Metadata is stored as a JSON string in the `lyrics` tag.
* `Slimmer Computed Latents`: falls back to the original, 12.9KiB way of storing latents (without the extra bits required for using the CVVP model).
* `Voice Fixer`: runs each generated audio clip through `voicefixer`, if available and installed.
* `Use CUDA for Voice Fixer`: allows voicefixer to use CUDA. Speeds up cleaning the output, but at the cost of more VRAM consumed. Disable if you OOM.
* `Do Not Load TTS On Startup`: skips loading TorToiSe on initialization, but will get loaded when anything that requires it needs it. This is useful if you're doing non-TTS functions that require VRAM, but you'll OOM while doing it when the model is loaded (for example, training).
* `Delete Non-Final Output`: if enabled and using multi-line generation, it will delete the individual pieces after combining. If enabled and using Voicefixer, it will remove the un-fixed file. Useful for reducing clutter.

* `Sample Batch Size`: sets the batch size when generating autoregressive samples. Bigger batches result in faster compute, at the cost of increased VRAM consumption. Leave to 0 to calculate a "best" fit.
* `Unsqueeze Sample Batches`: takes your batches of samples and unsqueezes them after sampling, effectively doing the CLVP pass at a batch size of 1. With it, you can safely increase your sample batch size without fear of OOMing during the CLVP pass.
* `Gradio Concurrency Count`: how many Gradio events the queue can process at once. Leave this over 1 if you want to modify settings in the UI that updates other settings while generating audio clips.
* `Auto-Calculate Voice Chunk Duration (in seconds)`: for automatically suggesting a voice chunk size, this value will divide the total duration of a voice's input samples. For example, 100 seconds worth of audio with this value as 10 will give 10 chunks. This is to make people stop shitting their pants when they OOM from not adjusting the `Voice Chunk` slider.
* `Output Volume`: adjusts the volume through amplitude scaling.
* `Device Override`: overrides the device name used to pass to PyTorch for hardware acceleration. You can use the accompanied `list_devices.py` script to map valid strings to GPU names. You can also pass `cpu` if you want to fallback to software mode.

* `TTS Backend`: the backend to target for training/inferencing. Defaults to [tortoise](https://git.ecker.tech/mrq/tortoise-tts/).
* `Autoregressive Model`: the autoregressive model to use for inference. This will look for models under `./models/finetunes/` and `./training/{voice}-finetune/models/`.
	- select "auto" to automatically select one based on the current voice loaded.
* `Diffusion Model`: the diffusion model used for inference. For now, this will only provide the default diffusion model, as you can override it manually with an argument flag or editing the `./config/exec.json` file.
* `Vocoder Model`: selects which vocoder to use. Univnet is the default vocoder, while BigVGAN is a better one.
* `Tokenizer JSON Path`: the tokenizer vocab. to use for tokenizing text input for training/inference. Selecting the provided `ipa.json` is experimental, only select it if you know what you are doing.

* `Refresh Model List`: updates the above dropdown with models
* `Check for Updates`: manually checks for an update for this rep.
* `(Re)Load TTS`: either initializes or reinitializes TorToiSe. You should not need to use this unless you change some settings, like Low VRAM.