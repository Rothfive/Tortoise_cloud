## Training / Finetuning

This tab will contain a collection of sub-tabs pertaining to training.

**!**NOTE**!**: this page needs heavy cleanup. Please keep this in mind.

### Pre-Requisites

Before continuing, please be sure you have an adequate GPU. The bare minimum requirements depend on your GPU's architecture, VRAM capacity, and OS.

* Windows + AMD: unfortunately, I cannot leverage DirectML to provide compatibility for training on Windows + AMD systems. If you're insistent on training, please use a Colab notebook.
* Linux + AMD:
	- a card with at least 6GiB of VRAM (with `bitsandbytes-rocm`)
    - a card with at least 12GiB of VRAM (without `bitsandbytes-rocm`)
* NVIDIA:
	- Pascal (10-series) and before: a card with at least 12GiB of VRAM.
    - Turing (20-series) and beyond: a card with at least 6GiB of VRAM.

Unfortunately, while the official documentation says Kepler-and-beyond cards are able to leverage some of BitsAndBytes' optimizations, in practice (at least on Windows), it seems it's only Turing-and-beyond. Some users have reported issues on Pascal GPUs, while others were able to get training to work with bitsandbytes, albeit with harmed performance.

To check if it works, you should see a message saying it is using BitsAndBytes optimizations on training startup.

### Finetune Capabilities

Training/finetuning a model offers a lot of improvements over using the base model. This can range from:
* better matching to a given voice's traits
	- for example, getting better David Hayter's Solid Snake
* capturing an accent to generate voice samples from
	- personally untested, but has been done
* teaching it an entire new language, like Japanese
	- personally tested on a dataset size of 14920 audio clips from a gacha I haven't played in ages, Japanese is replicated pretty decently

If any of the above is of interest, then you're on the right track.

## Prepare Dataset

This section will cover how to prepare a dataset for training.

* `Dataset Source`: a valid folder under `./voice/`, as if you were using it to generate with.
* `Language`: language code to transcribe to (leave blank to auto-deduce):
	- beware, as specifying the wrong language ***will*** let whisper translate it, which is ultimately pointless if you're trying to train aganst.
* `Validation Text Length Threshold`: transcription text lengths that are below this value are culled and placed in the validation dataset. Set 0 to ignore.
* `Validation Audio Length Threshold`: audio lengths that are below this value are culled and placed in the validation dataset. Set 0 to ignore.
* `Skip Already Transcribed`: skip transcribing a file if it's already processed and exists in the `whisper.json` file. Perfect if you're adding new files, and want to skip old files, while allowing you to re-transcribe files.
* `Slice Segments`: trims your source, based on the timestamps returned by whisper.
	- **!**NOTE**!**: please, please manually curate your segments. These aren't always accurate; sometimes it will trim too liberally.
* `Trim Silence`: leverages TorchAudio's VAD to trim out silence, reducing the actual size of the audio files, saving a little more processing time and VRAM consumption when training.
* `Slice Start Offset`: offsets the beginning timestamp when slicing audio files.
* `Slice End Offset`: offsets the end timestamp when slicing audio files.
* `Transcribe and Process`: begin transcription, while also slicing if necessary, and binning lines into either the validation or training datasets.
* `(Re)Slice Audio`: re-trims your source audios. Perfect if you did not prepare the dataset without slicing, or you modified the timestamps manually, and want to commit your changes.
* `(Re)Create Datasets`: re-parses the `whisper.json`, creating the files necessary for the training and, if requested, validation datasets.
* `Whisper Backend`: which Whisper backend to use. Currently supporting:
	- `openai/whisper`: the default, GPU backed implementation.
    - `lightmare/whispercpp`: an additional implementation. Leverages WhisperCPP with python bindings, lighter model sizes, and CPU backed.
    	+ **!**NOTE**!**: whispercpp is practically Linux only, as it requires a compiling environment that won't kick you in the balls like MSVC would on Windows.
* `Whisper Model`: whisper model to transcribe against. Larger models boast more accuracy, at the cost of longer processing time, and VRAM comsumption.
	- **!**NOTE**!**: the large model allegedly has problems with timestamps, moreso than the medium one.

This tab will leverage any voice you have under the `./voices/` folder, and transcribes your voice samples using [openai/whisper](https://github.com/openai/whisper) to prepare an LJSpeech-formatted dataset to train against.

It's not required to dedicate a small portion of your dataset for validation purposes, but it's recommended, as it helps remove data that's too small to be useful for. Using a validation dataset will help measure how well the finetune is at synthesizing speech from an input that it has not trained against.

If you're transcribing English text that's already stored as separate sound files (for example, one sentence per file), there isn't much of a concern with utilizing a larger whisper model, as transcription of English is already very decent with even the smaller models.

However, if you're transcribing something non-Latin (like Japanese), or need your source sliced into segments (if you have everything in one large file), then you should consider using a larger model for better timestamping (however, the large model seems to have some problems providing accurate segmentation).
* **!**NOTE**!**: be very careful with naively trusting how well the audio is segmented. Be sure to manually curate how well 

## Generate Configuration

This will generate the YAML necessary to feed into training. For documentation's sake, below are details for what each parameter does:
* `Epochs`: how many times you want training to loop through your data. This *should* be dependent on your dataset size, as I've had decent results with 500 epochs for a dataset size of about 60.
* `Learning Rate`: rate that determines how fast a model will "learn". Higher values train faster, but at the risk of frying the model, overfitting, or other problems. The default is "sane" enough for safety, especially in the scope of retraining, but definitely needs some adjustments. If you want faster training, bump this up to `0.0001` (1e-5), but be wary you may fry your finetune without tighter scheduling.
* `Text_CE LR Weight`: an experimental setting to govern how much weight to factor in with the provided learning rate. This is ***a highly experimental tunable***, and is only exposed so I don't need to edit it myself when testing it. ***Leave this to the default 0.01 unless you know what you are doing.***
	- In theory, for non-Latin languages, you should set this to 1.
* `Learning Rate Scheme`: sets the type of learning rate adjustments, each one exposes its own options:
	- `Multistep`: MultiStepLR, will decay at fixed intervals to by a factor (default set to 0.5, so it will halve every milestone).
		+ `Learning Rate Schedule`: a list of epochs on when to decay the learning rate. More experiments are needed to determine optimal schedules.
    - `Cos. Annealing`: CosineAnnealingLR_Restart, will gradually decay the learning rate over training, and restarts periodically
    	+ `Learning Rate Restarts`: how many times to "restart" the learning rate scheduling, but with a decay. 
* `Batch Size`: how large of a batch size for training. Larger batch sizes will result in faster training steps, but at the cost of increased VRAM consumption. Smaller batch sizes might not fully saturate your card's throughput, as well as offering *some* theoretical accuracy loss.
* `Gradient Accumulation Size` (*originally named `mega batch factor`*): This will further divide batches into mini-batches, parse them in sequence, but only updates the model after completing all mini-batches. This effectively saves more VRAM by de-facto running at a smaller batch size, while behaving as if running at a larger batch size. This does have some quirks at insane values, however.
* `Save Frequency`: how often to save a copy of the model during training in epochs. It seems the training will save a normal copy, an `ema` version of the model, *AND* a backup archive containing both to resume from. If you're training on a Colab with your Drive mounted, these can easily rack up and eat your allotted space. You *can* delete older copies from training, but it's wise not to in case you want to resume from an older state.
* `Validation Frequency`: governs how often to run validation.
* `Half-Precision`: setting this will convert the base model to float16 and train at half precision. This *might* be faster, but quality during generation *might* be hindered. I've trained against a small dataset (size 17) of Solid Snake for 3000 epochs, and it *works*, but you *must* enable Half-Precision for generation when using half-precision models. On CUDA systems, this is irrelevant, as everything is secretly trained using integer8 with bitsandbyte's optimizations.
* `BitsAndBytes`: specifies if you want to train with BitsAndBytes optimizations enabled. Enabling this makes the above setting redundant. You ***should*** really leave this enabled unless you absolutely are sure of what you're doing, as this is crucial to reduce VRAM usage.
* `Worker Processes`: tells the training script how many worker processes to spawn. I don't think more workers help training, as they just consume a lot more system RAM, especially when you're using multiple GPUs to train. 2 is sensible, so leave it there.
* `Source Model`: the source model to finetune against. With it, you can re-finetune already finetuned models (for example, taking a Japanese finetune that can speak Japanese well, but you want to refine it for a specific voice). You *should* leave this as the default autoregressive model unless you are sure of what you're doing. 
* `Resume State Path`: the last training state saved to resume from. The general path structure is what the placeholder value is. This will resume from whatever iterations it was last at, and iterate from there until the target step count (for example, resuming from iteration 2500, while requesting 5000 iterations, will iterate 2500 more times).
* `Dataset`: a dataset generated from the `Prepare Dataset` tab.

and, some buttons:
* `Refresh Dataset List`: updates the dataset list, required when new datasets are added
* `Reuse/Import Existing Dataset Settings`: pulls the settings used for a dataset. This will check for an existing training output first, before it checks the actual dataset in the `./training/` folder. This is primarily a shortcut for me when I'm testing settings.
* `Validate Training Configuration`: does some sanity checks to make sure that training won't throw an error, and offer suggested settings. You're free to change them after the fact, as validation is not done on save.
* `Save Training Configuration`: writes the settings to the training YAML, for loading with the training script.

After filling in the values, click `Save Training Configuration`, and it should print a message when it's done.

### Suggested Settings

If you're looking to quickly get something trained in under 100 epochs, these settings work decent enough. I've had three models quickly trained with these settings with astounding success, but one with moderate success.
* dataset size of <= 200:
	- Epochs: `100` (50 is usually "enough")
	- Learning Rate: `0.0001`
	- Learning Rate Scheme: `MultiStepLR`
	- Learning Rate Schedule: `[9, 18, 25, 33, 50, 59]` (or `[4, 9, 18, 25, 33, 50, 59]`, if you run into NaN issues)
    
However, if you want accuracy, I suggest an LR of 1e-5 (0.00001), as longer training at low LRs definitely make the best models.

The learning rate is definitely important, as:
* too large of a learning rate will have it very hard for it to land at the absolute minimum
* too little of a learning rate and it will either:
	- take much much longer to reach the absolute minimum, if in the right well
    - get stuck at the local minimum, and never can find the absolute minimum
    	+ LR restarts help alleviate this, as it will "shake" things out of any local-but-not-absolute minimums.

### Resuming Training

You can easily resume from a previous training state within the web UI as well.
* select the `Dataset` you want to resume from
* click `Import Dataset`
* it'll pull up the last used settings and grab the last saved state to resume from
	- you're free to adjust settings, like epoch counts, and validation/save frequencies.
    - **!**NOTE**!**: ensure your dataset size and batch sizes match, as things will get botched if you don't
* click `Save Training Setting`
	- you're free to revalidate your settings, but it shouldn't be necessary if you changed nothing
And you should be good to resume your training.

I've done this plenty of times and haven't had anything nuked or erased. As a safety precaution, DLAS will always move the existing folder as a backup if it's starting from a new training and not resuming. If it resumes, it won't do that, and nothing should be overwritten.

In the future, I'll adjust the "resume state" to provide a dropdown instead when selecting a dataset, rather than requiring to import and deduce the most recent state, to make things easier.

**!**NOTE**!**: due to some quirks when resuming, training will act funny for a few epochs, at least when using multi-GPUs. This probably has to do with the optimizer state not quite being recovered when resuming, so the losses will keep oscillating.

### Changing Base Model

In addition to finetuning the base model, you can specify what model you want to finetune, effectively finetuning finetunes. Theoretically, this is useful for finetuning off of language models to fit a specific voice.

I have not got a chance to test this, as I mistakenly deleted my Japanese model, and I've been having bad luck getting it re-trained.

## Run Training

After preparing your dataset and configuration file, you are ready to train. Simply select a generated configuration file, click train, then keep an eye on either the console window to the right for output, or console output in your terminal/command prompt.

If you check `Verbose Console Output`, *all* output from the training process gets forwarded to the console window on the right until training starts. This is useful if you access to your terminal output is an inconvenience.

If you bump up the `Keep X Previous States` above 0, it will keep the last X number of saved models and training states, and clean up the rest on training start, and every save.

If everything is done right, you'll see a progress bar and some helpful metrics. Below that, is a graph of the loss rates:
* `current epoch / total epochs`: how far along you are in terms of epochs
* `current iteration / total iterations`: how far along you are in terms of iterations
* `current batch / total batches`: how far along you are within an epoch
* `iteration throughput rate`: the time it took to process the last iteration
* `ETA`: estimated time to completion; will use the iteration throughput rate to estimate
	- **!**NOTE**!**: this is currently broken, I'm not sure what broke it
* `Loss`: the last reported loss value
* `LR`: the last reported learning rate
* `Next milestone in:` reports the next "milestone" for training, and how many more iterations left to reach it.
	- **!**NOTE**!**: this is currently broken too.

Metrics like loss rates and learning rates get reported back after every iteration. This is useful to see how "ready" your model/finetune is.

If something goes wrong, please consult the output, as, more than likely, you've ran out of memory. On systems where VRAM is really tight, it seems pretty easy for it to get upset and OOM.

After you're done, the process will close itself, and you are now free to use the newly baked model. You can then head on over to the `Settings` tab, reload the model listings, and select your newly trained model in the `Autoregressive Model` dropdown, or utilize the `auto` feature and it should be deduced when using the voice with the same name as the dataset you trained it on.

### Training Graphs

#### Brief Explanation

To the right are two graphs to give a raw and rough idea on how well the model is trained.

The first graph will show an aggregate of loss values, and if requested, validation loss rates. These values quantify how much the output from the model deviates from the input sources. There's no one-size-fits-all value on when it's "good enough", as some models work fine with a high enough value, while some other models definitely benefit from really, low values. However, achieving a low loss isn't easy, as it's mostly predicated on an adequate learning rate.

The second graph isn't as important, but it models where the learning rate is at the last reported moment. 

Typically, a "good model" has the text-loss a higher than the mel-loss, and the total-loss a little bit above the mel-loss. If your mel-loss is above the text-loss, don't expect decent output. I believe I haven't had anything decent come out of a model with this behavior.

#### Slightly In-Depth

The autoregressive model predicts tokens in as `<speech conditioning>:<text tokens>:<MEL tokens>` string, where:
* speech conditioning is a vector representing a voice's latents
* text tokens (I believe) represents phonemes, which can be compared against the CLVP for "most likely candidates"
* MEL tokens represent the actual speech, which gets later converted to a waveform

Each curve is responsible for quantifying how accurate the model is.
* the text loss quantifies how well the predicted text tokens match the source text. This doesn't necessarily need to have too low of a loss. In fact, trainings that have it lower than the mel loss turns out unusuable.
* the mel loss quantifies how well the predicted speech tokens match the source audio. This definitely seems to benefit from low loss rates.
* the total loss is a bit irrelevant, and I should probably hide it since it almost always follows the mel loss, due to how the text loss gets weighed.

There's also the validation versions of the text and mel losses, which quantifies the defacto similarity from the generated output to the source output, as the validation dataset serves as outside data (as if you're normally generating something). If there's a large deviation betweent he reported losses and the validation losses, then your model probably has started to overfit for the source material.

Below all of that is the learning rate graph, which helps to show what the current learning rate is at. It's not a huge indicator of how training is, as the learning rate curve is determinative.

### Training Validation

In addition, the training script also allows for validating your model against a separate dataset, to see how well it performs when using data it's not trained on.

However, these metrics are not re-incorporated into the training, as that's not what the validation dataset is for.

**!**NOTE**!**: Validation iterations sometimes counts towards normal iterations, for whatever reason.

### Multi-GPU Training

**!**NOTE**!**: This has only been tested on Linux. Windows lacks `nccl` support, but apparently works under WSL2.

If you have multiple GPUs, you can easily leverage them by simply specifying how many GPUs you have when generating your training configuration. With it, it'll divide out the workload by splitting the batches to work on among the pool (at least, to my understanding).

**!**NOTE**!**: However, the larger the dataset, the more apparent problems will surface:
* single-GPU training setups shuffles around the order of the source files which really helps with training. This is disabled when using multiple GPUs, effectively harming accuracy a slightly noticeable amount.
* optimizer states don't seem to properly get restored when resuming. Losses will oscillate for a while when resuming while training on multiple GPUs.
* throughput increase is a little finnicky. I found not using gradient accumulation (ga=1) helps increase throughput.

I imagine I'm probably neglecting some settings that would alleviate my headaches. Only use multiple GPU training if you're like me and incidentally have matching GPUs.

### Training Output

Due to the nature of the interfacing with training, some discrepancies may occur:
* the training script calculates what an epoch is slightly different than what the UI calculates an epoch as. This might be due to how it determines what lines in the dataset gets culled out from non-evenly-divisible dataset sizes by batch sizes. For example, it might think a given amount of iterations will fill 99 epochs instead of 100.
* for long, long generations on a publicly-facing Gradio instance (using `share=True`), the UI may disconnect from the program. This can be remedied using the `Reconnect` button, but the UI will appear to update every other iteration. This is because it's still trying to "update" the initial connection, and it'll grab the line of output from stdio, and will alternate between the two sessions.