## Sourcing

Now that the easy part of setting up is done, it's time for the hard part: sourcing your voice samples.

As this fork was worked on entirely in the scope of videogame characters, the sources below pertain mostly to that. There might be some overlap, given some of these are sourced from other /v/irgins, so you might as well take a peak in case your not-videogames voice is in there:

* [Sounds Resource](https://www.sounds-resource.com/) has a plethora of sound content, including voice lines.
* [Silent Hill Media](http://silenthillmedia.net/home.htm) has voice lines for Silent Hill characters, albeit given the nature, some of them might be mixed together (for example, exchanges between Maria and James).
* [the AIVoiceStuff rentry from /v/](https://rentry.org/AIVoiceStuff) during the prime of AI voice cloning threads there has a good repository of sources, not just for vidya. However, since that rentry is scoped around 11.AI, the provided output examples are from 11.AI. I suppose you can use those to compare against TorToiSe's quality. Given cloning has been removed from free tier (and the threads pretty much died down), that rentry isn't getting updated.
* [a mega.nz collection from /v/](https://mega.nz/folder/AHtCyYRa#WoWv9ug6vg27XfXOjfga-Q), also during the prime of AI voice cloning threads, also have a decent source of samples, not just for vidya. 

**!**NOTE**!**: I'm merely providing easy directions to acquire sources.

## Preparing

Now that the tough part is dealt with, it's time to prepare voice clips to use.

Unlike training embeddings for AI image generations, preparing a "dataset" for voice cloning is very simple.

As a general rule of thumb, try to source clips that aren't noisy, solely the subject you are trying to clone, and doesn't contain any non-words (like yells, guttural noises, etc.). If you must, run your source through a background music/noise remover (how to is an exercise left to the reader). It isn't entirely a detriment if you're unable to provide clean audio, however. Just be wary that you might have some headaches with getting acceptable output.

Nine times out of ten, you should be fine using as many clips as possible. There's (now) no preference between combining your audio into one file, or leaving it split. However, if you're aiming for a specific delivery, it *should* be best for you to narrow down to just using that as your provided source (for example, changing one word in a line).

There's no hard specifics on how many, or how long, your sources should be.

If you're looking to trim your clips, in my opinion, ~~Audacity~~ Tenacity works good enough. Power users with FFMPEG already installed can simply used the provided conversion script in `.\tortoise\convert\`.

After preparing your clips as WAVs, you can now add in your new voice source:
* open up the `.\ai-voice-cloning\voices\` folder
* create a new folder in whatever name you want
* dump your clips into that folder