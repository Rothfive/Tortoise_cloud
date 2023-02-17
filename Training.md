## Training / Finetuning

This tab will contain a collection of sub-tabs pertaining to training.


### Pre-Requisites

Before continuing, please be sure you have an adequate GPU. The bare minimum requirements depend on your GPU's architecture, VRAM capacity, and OS.

* Windows + AMD: unfortunately, I cannot leverage DirectML to provide compatibility for training on Windows + AMD systems. If you're insistent on training, please use a Colab notebook.
* Linux + AMD: (**!**UNVERIFIED**!**)
	- a card with at least 16GiB of VRAM (without bitsandbytes)
    - (theoretically) a card with at least 6GiB of VRAM, with [broncotc/bitsandbytes-rocm](https://github.com/broncotc/bitsandbytes-rocm)
* NVIDIA:
	- Pascal (10-series) and before: a card with at least 16GiB of VRAM.
    - Turing (20-series) and beyond: a card with at least 6GiB of VRAM.

Unfortunately, only Turing cards (and beyond) have the necessary dedicated silicon to do integer8 calculations, an optimization leveraged by BitsAndBytes to allow for even the low-end consumer cards to train. However, BitsAndByte's documentation says this restriction is only for inferencing, and instead, the real requirement is Kepler and beyond. Unfortunately, I have no real way to test this, as it seems users with Kepler/Pascal cards are getting esoteric CUDA errors when using BitsAndBytes.

If you're on Windows and using an installation of this software from before 2023.02.24, and you want to (and can) use BitsAndBytes, please consult https://git.ecker.tech/mrq/ai-voice-cloning/issues/25 for a simple guide to copying the right files.

If you're on Windows using an installation after 2023.02.24, the setup should have already taken care of copying the necessary files to use BitsAndBytes.

To check if it works, you should see a message saying it is using BitsAndBytes optimizations on training startup.

### Capabilities

Training/finetuning a model offers a lot of improvements over using the base model. This can range from:
* better matching to a given voice's traits
	- for example, getting better David Hayter's Solid Snake
* capturing an accent to generate voice samples from
	- personally untested, but has been done
* teaching it an entire new language, like Japanese
	- personally tested on a dataset size of 14920 audio clips from a gacha I haven't played in ages, Japanese is replicated pretty decently

If any of the above is of interest, then you're on the right track.

## Prepare Dataset

This section will aid in preparing the dataset for fine-tuning.

Dataset sizes can range from a few sentences, to a large collection of lines. However, do note that smalller dataset sizes require more epochs to finetune against, as there's less iterations invested to train per epoch.

Simply put your voice sources in its own folder under `./voices/` (as you normally would when using a voice for generaton), specify the language to transcribe to (default: English), then click Prepare.

This utility will leverage [openai/whisper](https://github.com/openai/whisper/) to transcribe the audio. Then, it'll slice the audio into pieces that the transcription found fit. Afterwards, it'll output this transcript as an LJSpeech-formatted text file: `train.txt`.

As whisper uses `fffmpeg` to handle it's audio processing, you must have a copy of `ffmpeg` exposed and accessible through your PATH environment variable. On Linux, this is simply having it installed through your package manager. On Windows, you can just download a copy of `ffmeg.exe` and drop it into the `./bin/` folder.

Transcription is not perfect, however. Be sure to manually quality check the outputted transcription, and edit any errors it might face. For things like Japanese, it's expected for things that would be spoken katakana to be coerced into kanji. In addition, when generating a finetuned model trained on Japanese:
* some kanji might get coerced into the wrong pronunciation.
* small kana like the `っ` of `あたしって` gets coerced as the normal kana.
* some punctuation like `、` may prematurely terminate a sentence.

**!**NOTE**!**: you might get some funky errors; consult this [issue](Issues#user-content-attributeerror-module-ffmpeg-has-no-attribute-input) if you do.

## Generate Configuration

This will generate the YAML necessary to feed into training. Here, you can set some parameters on how training will be done:
* `Epochs`: how many times you want training to loop through your data. This *should* be dependent on your dataset size, as I've had decent results with 500 epochs for a dataset size of about 60.
* `Learning Rate`: rate that determines how fast a model will "learn". Higher values train faster, but at the risk of frying the model, overfitting, or other problems. The default is "sane" enough for safety, especially in the scope of retraining, but definitely needs some adjustments. If you want faster training, bump this up to `0.0001` (1e-5), but be wary you may fry your finetune without tighter scheduling.
* `Text_CE LR Weight`: an experimental setting to govern how much weight to factor in with the provided learning rate. This is ***a highly experimental tunable***, and is only exposed so I don't need to edit it myself when testing it. ***Leave this to the default 0.01 unless you know what you are doing.***
* `Learning Rate Schedule`: a list of epochs on when to decay the learning rate. You really should leave this as the default.
* `Batch Size`: how large of a batch size for training. Larger batch sizes will result in faster training steps, but at the cost of increased VRAM consumption. This value must exceed the size of your dataset, and *should* be evenly divisible by your dataset size.
* `Mega Batch Factor`: According to the documentation, `DLAS also supports "mega batching", where multiple forward passes contribute to a single backward pass`. If you can spare the VRAM, I suppose you can bump this to 8. If you're pressed for VRAM, you can lower this down to 1. If you have really small batch sizes, use what the validator gives out.
* `Print Frequency`: how often the trainer should print its training statistics in epochs. Printing takes a little bit of time, but it's a nice way to gauge how a finetune is baking, as it lists your losses and other statistics. This is purely for debugging and babysitting if a model is being trained adequately. The web UI *should* parse the information from stdout and grab the total loss and report it back.
* `Save Frequency`: how often to save a copy of the model during training in epochs. It seems the training will save a normal copy, an `ema` version of the model, *AND* a backup archive containing both to resume from. If you're training on a Colab with your Drive mounted, these can easily rack up and eat your allotted space. You *can* delete older copies from training, but it's wise not to in case you want to resume from an older state.
* `Resume State Path`: the last training state saved to resume from. The general path structure is what the placeholder value is. This will resume from whatever iterations it was last at, and iterate from there until the target step count (for example, resuming from iteration 2500, while requesting 5000 iterations, will iterate 2500 more times).
* `Half-Precision`: setting this will convert the base model to float16 and train at half precision. This *might* be faster, but quality during generation *might* be hindered. I've trained against a small dataset (size 17) of Solid Snake for 3000 epochs, and it *works*, but you *must* enable Half-Precision for generation when using half-precision models. On CUDA systems, this is irrelevant, as everything is secretly trained using integer8 with bitsandbyte's optimizations.
* `BitsAndBytes`: specifies if you want to train with BitsAndBytes optimizations enabled. Enabling this makes the above setting redundant. You ***should*** really leave this enabled unless you absolutely are sure of what you're doing, as this is crucial to reduce VRAM usage.
* `Source Model`: the source model to finetune against. With it, you can re-finetune already finetuned models (for example, taking a Japanese finetune that can speak Japanese well, but you want to refine it for a specific voice). You *should* leave this as the default autoregressive model unless you are sure of what you're doing. 
* `Dataset`: a dataset generated from the `Prepare Dataset` tab.
and, some buttons:
* `Refresh Dataset List`: updates the dataset list, required when new datasets are added
* `Import Existing Dataset Settings`: pulls the settings used for a dataset. This will check for an existing training output first, before it checks the actual dataset in the `./training/` folder. This is primarily a shortcut for me when I'm testing settings.
* `Validate Training Configuration`: does some sanity checks to make sure that training won't throw an error, and offer suggested settings. You're free to change them after the fact, as validation is not done on save.
* `Save Training Configuration`: writes the settings to the training YAML, for loading with the training script.

After filling in the values, click `Save Training Configuration`, and it should print a message when it's done.

### Resuming Training

You can easily resume from a previous training state within the web UI as well.
* select the `Dataset` you want to resume from
* click `Import Dataset`
* it'll pull up the last used settings and grab the last saved state to resume from
	- feel free to adjust any other settings, like increasing the epoch count
	- **!**NOTE**!**: sometimes-but-not-all-the-time, the numbers might be a bit mismatched, due to some rounding errors when converting back from iterations as a unit to epochs as a unit
* click `Save Training Setting`
	- you're free to revalidate your settings, but it shouldn't be necessary if you changed nothing
And you should be good to resume your training.

I've done this plenty of times and haven't had anything nuked or erased. As a safety precaution, DLAS will always move the existing folder as a backup if it's starting from a new training and not resuming. If it resumes, it won't do that, and nothing should be overwritten.

In the future, I'll adjust the "resume state" to provide a dropdown instead when selecting a dataset, rather than requiring to import and deduce the most recent state, to make things easier.

### Changing Base Model

Currently in the web UI, there's no way to specify picking a different model (such as, using a finetune to train from). You must manually edit the `train.yaml` by specifying the path to the model you want to fine tune at line 117.

I have not tested if this is feasible, but I have tested that you can finetune from a model you have already finetuned from. For example, if you were to train a large dataset for a different language (Japanese), but you also want to finetune for a specific voice, you can re-finetune the Japanese model.

## Run Training

After preparing your dataset and configuration file, you are ready to train. Simply select a generated configuration file, click train, then keep an eye on either the console window to the right for output, or console output in your terminal/command prompt.

If you check `Verbose Console Output`, *all* output from the training process gets forwarded to the console window on the right. This output is buffered, up to the `Console Buffer Size` specified (for example, the last eight lines if 8).

If you bump up the `Keep X Previous States` above 0, it will keep the last X number of saved models and training states, and clean up the rest on training start, and every save. **!**NOTE**!** I did not extensively test this, only on test data, and it did not nuke my saves. I don't expect it to happen, but be wary.

If everything is done right, you'll see a progress bar and some helpful metrics. Below that, is a graph of the total GPT loss rate. 

After every `print rate` iterations, the loss rate will update and get reported back to you. This will update the graph below with the current loss rate. This is useful to see how "ready" your model/finetune is. The general rule of thumb is the lower, the better. I used to swear by values around `0.15` and `0.1`, but I've had nicer results when it's lower. But be wary, as this *may* be grounds for overfitment, as is the usual problem with training/finetuning.

If something goes wrong, please consult the output, as, more than likely, you've ran out of memory.

After you're done, the process will close itself, and you are now free to use the newly baked model.

You can then head on over to the `Settings` tab, reload the model listings, and select your newly trained model in the `Autoregressive Model` dropdown.

### Training Output

Due to the nature of the interfacing with training, some discrepancies may occur:
* the UI bases its units in epochs, and converts to the unit the training script bases itself in: iterations. Some slight rounding errors may occur. For example, at the last epoch, it might save one iteration before how many iterations given to train.
* the training script calculates what an epoch is slightly different than what the UI calculates an epoch as. This might be due to how it determines what lines in the dataset gets culled out from non-evenly-divisible dataset sizes by batch sizes. For example, it might think a given amount of iterations will fill 99 epochs instead of 100.
* because I have to reparse the training output, some statistics may seem a little inconsistent. For example, the ETA is extrapolated by the last delta between iterations. I could do better ways for this (like by delta time between epochs, averaging delta time between iterations and extrapolating from it, etc.).
* for long, long generations on a publically-facing Gradio instance (using `share=True`), the UI may disconnect from the program. This can be remedied using the `Reconnect` button, but the UI will appear to update every other iteration. This is because it's still trying to "update" the initial connection, and it'll grab the line of output from stdio, and will alternate between the two sessions.