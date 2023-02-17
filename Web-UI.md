## Using the Software

Now you're ready to generate clips. With the command prompt still open, simply enter `start.bat` (or `start.sh`), and wait for it to print out a URL to open in your browser, something like `http://127.0.0.1:7860`.

If you're looking to access your copy of TorToiSe from outside your local network, tick the `Public Share Gradio` button in the `Settings` tab, then restart.

Before actually using the software, please consult the [Collecting Samples](Collecting-Samples) page, as use of this software is under the heavy assumption you're using it to voice clone, rather than just synthesis voice. You can still use it with the `random` voice feature, but it's not the purpose.

TorToiSe, the underlying TTS software, is a zero-shot speech synthesizer: no further training is required, and some voices can get pretty decent output as-is with the default model.

However, some voices (or languages) require some fine-tuning of the base model to get better, stronger output. If you're looking to get better output, consider finetuning with the [Training](Training) tab.

If you're not sure, it does not hurt to play around with the default models, and see what works. 

For more information for a given tab, consult the Sidebar.