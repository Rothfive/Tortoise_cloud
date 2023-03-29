## Utilities

In this tab, you can find some helper utilities that might be of assistance.

### Import/Analyze

For now, an analog to the PNG info found in Voldy's Stable Diffusion Web UI resides here. With it, you can upload an audio file generated with this web UI to view the settings used to generate that output. Additionally, the voice latents used to generate the uploaded audio clip can be extracted.

If you want to reuse its generation settings, simply click `Copy Settings`.

To import a voice, click `Import Voice`. Remember to click `Refresh Voice List` in the `Generate` panel afterwards, if it's a new voice.

### Tokenizer

This section is for debugging purposes while working with custom tokenizer vocabs. You can use this to see how a given input will tokenize.

### Model Mixing

Given `model A` and `model B`, you can merge two models together by a `weight` factor (0 will favor `model A`, 1 will favor `model B`). My very, very, impromptu and cursory test yielded already great results:
* given finetunes for Melina (Elden Ring) and Mitsuru (Persona 3) with an equal weight of 0.5:
	- Mitsuru (Persona 3): https://vocaroo.com/16Fm7lzKFNoD
	- Melina (Elden Ring): https://vocaroo.com/1gH7PxosRjUr

Mixing is pretty quick, and will output the new model at `./models/finetunes/modelA_modelB_weight_merge.pth`

Use cases for this will be noted at a later time.