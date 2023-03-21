# AI Voice Cloning

This [repo](https://git.ecker.tech/mrq/ai-voice-cloning)/[rentry](https://rentry.org/AI-Voice-Cloning/) aims to serve as both a foolproof guide for setting up AI voice cloning tools for legitimate, local use on Windows/Linux, as well as a stepping stone for anons that genuinely want to play around with [TorToiSe](https://github.com/neonbjb/tortoise-tts).

Similar to my own findings for Stable Diffusion image generation, this rentry may appear a little disheveled as I note my new findings with TorToiSe. Please keep this in mind if the guide seems to shift a bit or sound confusing.

>\>Ugh... why bother when I can just abuse 11.AI?

You're more than welcome to, but TorToiSe is shaping up to be a very promising tool, especially with finetuning being feasible.

This is not endorsed by [neonbjb](https://github.com/neonbjb/). I do not expect this to run into any ethical issues, as it seems (like me), this is mostly for making funny haha vidya characters say funny lines.

## Glossary

To try and keep the terminology used here (somewhat) consistent and coherent, below are a list of terms, and their definitions (or at least, the way I'm using them):
* `voice cloning`: synthesizing speech to accurately replicate a subject's voice.
* `input clips` / `voice clips` / `audio input` / `voice samples` : the original voice source of the subject you're trying to clone.
* `waveform`: the raw audio.
* `sampling rate`: the bandwidth of a given waveform, represented as twice the frequency of the waveform it represents.
* `voice latents` / `conditional latents` / `latents`: computated traits of a voice.
* `autoregressive samples` (`samples` / `tokens`): the initial generation pass to output tokens, and (usually) the most computationally expensive. More samples = better "cloning", as you're generating more candidates to find the best sample.
* `CLVP`: Contrastive Language-Voice Pretraining: an analog to CLIP, but for voices. After the autoregressive samples pass, those samples/tokens are compared against the CLVP to find the best candidates.
* `CVVP`: Contrastive Voice-Voice Pretraining: a (deprecated) model that can be used weighted in junction with the CLVP.
* `candidates`: results from the comparing against the CLVP/CVVP models. (Assumed to be) ordered from best to worst.
* `diffusion decoder` / `vocoder`: these passes are responsible for encoding the tokens into a MEL spectrogram into a waveform.
* `diffusion iterations`: how many passes to put into generating the output waveform. More iterations = better audio quality.
* `diffusion sampler` / `sampler`: the sampling method used during the diffusion decoding pass, albeit a bit of a misnomer. Currently, only two samplers are implemented.

* `training` / `finetuning`: re-training the model to learn better traits for a given use-case, be it voice traits, accents, or even languages.
* `dataset`: an LJSpeech formatted text file with transcriptions assigned to voice files
* `learning rate`: the rate of change when training a model. Lower values are safer, but require much time. Higher values get change faster, at the risk of frying a model.
* `epoch`: a unit related to training, equal to the number of iterations to complete one entire pass over your dataset.
* `iteration` / `step`: a step to complete one batch size worth of training
* `learning rate schedule`: a list of epoch/iteration points to decay the learning rate
* `half precision`: trains at a lower precision to (theoretically) reduce VRAM consumption and increase throughput. Entirely useless when using...
* `bitsandbytes optimizations`: trains at a lower precision (integer8) by leveraging dedicated silicon and other optimizations to achieve large reductions in VRAM consumption. Arguably less stable, due to the nature of quantizing, but the downside is negligible compared to what it offers.
* `loss rate`: a value to measure the deviation from a model's generated output from the source output. Lower values = better, but too low of a value results in overfitment, and effectively botches your model, as extrapolation is tougher. I'd argue a value around `0.1` is fine, as it works for Textual Inversion, but I'm having second doubts. Just play around with different iterations of your models.

## Modifications

My fork boasts the following additions, fixes, and optimizations:
* a competent web UI made in Gradio to expose a lot of tunables and options
* cleaned up output structure of resulting audio files
* caching computed conditional latents for faster re-runs
	- additionally, regenerating them if the script detects they're out of date
* uses the entire audio sample instead of the first four seconds of each sound file for better reproducing
* activated unused DDIM sampler
* use of some optimizations like `kv_cache`ing for the autoregression sample pass, and keeping data on GPU 
* compatibility with DirectML
* easy install scripts
* integrated training
* very simple training configuration tool
* LJSpeech-formatted dataset creation
* leverages bitsandbytes for training/finetuning, and (albeit unnoticeable) for inferencing
* and more!