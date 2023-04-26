## Generate

### Quick Usage

If you're easily overwhelmed with the below section, fear not, as it's very simple. Just:
* input your text
* select your voice
* click `Ultra Fast`
* click `Generate`

If you're not too happy with the outputted result, you're free to play around with the below settings to dial in what you want.

### Detailed Usage

You'll be presented with a bunch of options in the default `Generate` tab, but do not be overwhelmed, as most of the defaults are sane, but below are a rough explanation on which input does what:
* `Prompt`: text you want to be read. You wrap text in `[brackets]` for "prompt engineering", where it'll affect the output, but those words won't actually be read.
* `Line Delimiter`: String to split the prompt into pieces. The stitched clip will be stored as `combined.wav`. To split by a new line, enter `\n`.
* `Emotion`: the "emotion" used for the delivery. This is a shortcut to utilizing "prompt engineering" by starting with `[I am really <emotion>,]` in your prompt. This is merely a suggestion, not a guarantee.
* `Custom Emotion + Prompt` (if `Custom` is selected): a non-preset "emotion" used for the delivery. This is a shortcut to utilizing "prompt engineering" by starting with `[<emotion>]` in your prompt.
* `Voice`: the voice you want to clone. You can select `microphone` if you want to use input from your microphone. You can also use `random` for it to use a randomly generated voice.
* `Microphone Source`: Use your own voice from a line-in source.
* `Voice Chunks`: a slider to determine how many pieces to process your voice dataset when computing conditional latents. The lower the number, the bigger the pieces and the more VRAM is needed. The smaller the pieces, the less VRAM is needed, but the more likely you will slice mid-phoneme. Playing around with this will most definitely affect the output of your cloning, as some datasets will work better with different values.
	- a "safe" value will be automatically calculated based on the total duration of your voice input samples, to avoid blindly OOMing. You can adjust this factor in `Settings`.
    - if you've created an LJSpeech dataset (Under `Training` > `Prepare Dataset`), this will automatically set to 0, hinting for the routine to use the dataset audio and padding them to a common size, for a little more accurate capturing of the latents.
* `Refresh Voice List`: updates the voice list
* `(Re)Compute Voice Latents`: (re)computes the conditional latents for a given voice.

Below are generation settings, which affect the technical aspects of how your inputs are processed:
* `Candidates`: number of outputs to generate, starting from the best candidate. Depending on your iteration steps, generating the final sound files could be cheap, but they only offer alternatives to the samples generated to pull from (in other words, the later candidates perform worse), so don't be compelled to generate a ton of candidates.
* `Seed`: initializes the PRNG to this value. Use this if you want to reproduce a generated voice.
* `Preset`: shortcut values for sample count and iteration steps. Clicking a preset will update its corresponding values. Higher presets result in better quality at the cost of computation time.
* `Samples`: analogous to samples in image generation. More samples = better resemblance / clone quality, at the cost of performance. This strictly affects clone quality.
* `Iterations`: influences audio sound quality in the final output. More iterations = higher quality sound. This step is relatively cheap, so do not be discouraged from increasing this. This strictly affects quality in the actual sound.
* `Temperature`: how much randomness to introduce to the generated samples. Lower values = better resemblance to the source samples, but some temperature is still required for great output.
	- **!**NOTE**!**: This value is very inconsistent and entirely depends on the input voice. In other words, some voices will be receptive to playing with this value, while others won't make much of a difference.
	- **!**NOTE**!**: some voices will be very receptive to this, where it speaks slowly at low temperatures, but nudging it a hair and it speaks too fast.
    - **!**NOTE**!**: this appears to "replace" larger autoregressive sample sizes, for example, 16 samples to 0.8 temperature yields similar results to something like 64 to 0.4 (roughly).

Below are an explanation of experimental flags. Messing with these might impact performance, as these are exposed only if you know what you are doing.
* `Half-Precision`: (attempts to) hint to PyTorch to auto-cast to float16 (half precision) for compute. Disabled by default, due to it making computations slower. With BitsAndBytes optimizations, this seems to offer a little boost in subjective quality. This is required if using a model at half-precision.
* `Conditional Free`: a quality boosting improvement at the cost of some performance. Enabled by default, as I think the penaly is negligible in the end.
* `Pause Size`: Governs how large pauses are at the end of a clip (in token size, not seconds). Increase this if your output gets cut off at the end.
	- **!**NOTE**!**: sometimes this is merely a suggestion and not a guarantee. Some generations will be sensitive to this, while others will not.
	- **!**NOTE**!**: too large of a pause size can lead to unexpected behavior.
* `Diffusion Sampler`: sampler method during the diffusion pass. Currently, only `P` and `DDIM` are added, but does not seem to offer any substantial differences in my short tests.
	`P` refers to the default, vanilla sampling method in `diffusion.py`.
	To reiterate, this ***only*** is useful for the diffusion decoding path, after the autoregressive outputs are generated.
* `CVVP Weight`: governs how much weight the CVVP model should influence candidates. The original documentation mentions this is deprecated as it does not really influence things, but you're still free to play around with it.
	Currently, setting requires regenerating your voice latents, as I forgot to have it return some extra data that weighing against the CVVP model uses. Oops.
	Setting this to 1 leads to bad behavior.
* `Top P`: P value used in nucleus sampling; lower values mean the decoder produces more "likely" (aka boring) outputs.
* `Diffusion Temperature`: the variance of the noise fed into the diffusion model; values at 0 are the "mean" prediction of the diffusion network and will sound bland and smeared.
* `Length Penalty`: a length penalty applied to the autoregressive decoder; higher settings causes the model to produce more terse outputs.
* `Repetition Penalty`: a penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc.
* `Conditioning-Free K`: determines balancing the conditioning free signal with the conditioning-present signal. 

After you fill everything out, click `Run`, and wait for your output in the output window. The sampled voice is also returned, but if you're using multiple files, it'll return the first file, rather than a combined file.

All outputs are saved under `./result/[voice name]/`. On some browsers, you're able to directly download the file with the three-dot menu in the HTML5 audio element.

To save you from headaches, I strongly recommend playing around with shorter sentences first to find the right values for the voice you're using before generating longer sentences.

### `random` voice

Due to how the entire generation pass is handles, the random voice will very, very, very much be random, even with the same latents. If you want a persistent random voice, you're free to manually "loopback" the input, by treating the generated output as a new voice.

### Prompt Setting Editing

If you want to procedurally edit any generation settings (for example, switch between voices), you can add to the start of the line a JSON string containing the settings you want to override. For example:
```
{"voice": "james"} Is that really you, Mary?
{"voice": "maria", "autoregressive_model": "auto" } The name is Maria.
```
will load the `james` voice using the currently loaded model for the first line, while the second line will load the `maria` voice and attempt to search for a valid model (will check for `./models/finetunes/{voice}.pth` first, then the latest model in `./training/{voice}-finetune/models/#_gpt.pth`).

#### Available Settings

You can always refer to a list of valid settings under [`./src/utils.py:155`](https://git.ecker.tech/mrq/ai-voice-cloning/src/branch/master/src/utils.py#L155), but for ease of access:
* `temperature`
* `top_p`
* `diffusion_temperature`
* `length_penalty`
* `repetition_penalty`
* `cond_free_k`
* `num_autoregressive_samples`
* `sample_batch_size`
* `diffusion_iterations`
* `voice_samples`
* `conditioning_latents`
* `use_deterministic_seed`
* `return_deterministic_state`
* `k`
* `diffusion_sampler`
* `breathing_room`
* `progress`
* `half_p`
* `cond_free`
* `cvvp_amount`
* `autoregressive_model`

You can set `autoregressive_model` to `auto` alongside a `voice` to automatically deduce a model for a given voice.

### Additional Backends

Some additional backends have not-as-tight integration. You can specify a backend by passing `--tts-backend="backend name"` when launching the starting script. These backends are ***highly*** experimental, so a lot of features that work for TorToiSe are unavailable for these backends. I tried my best to hide what's not needed, but I'm not perfect.

* `vall-e`: Utilizes [mrq/vall-e](https://git.ecker.tech/mrq/vall-e) as the TTS backend
    + Requires the repo to be cloned to `./modules/vall-e/` and installed using `pip3 install -e ./modules/vall-e/`
* `bark`: Utilizes [suno-ai/bark/](https://github.com/suno-ai/bark/) as the TTS backend
    + Requires the repo to be cloned to `./modules/bark/` and installed using `pip3 install -e ./modules/bark/`
    + Requires the target voice to have already been transcribed using the `Training > Prepare Dataset` tab.
        - This is because creating custom speaker prompts require a transcription of the reference prompt, and what better way to do that than leverage existing transcriptions.