# AI Toolkit by Ostris

## IMPORTANT NOTE - READ THIS
This is an active WIP repo that is not ready for others to use. And definitely not ready for non developers to use.
I am making major breaking changes and pushing straight to master until I have it in a planned state. I have big changes
planned for config files and the general structure. I may change how training works entirely. You are welcome to use
but keep that in mind. If more people start to use it, I will follow better branch checkout standards, but for now
this is my personal active experiment.

Report bugs as you find them, but not knowing how to train ML models, setup an environment, or use python is not a bug.
I will make all of this more user-friendly eventually

I will make a better readme later.

## Installation

Requirements:
- python >3.10
- Nvidia GPU with enough ram to do what you need
- python venv
- git



Linux:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python3 -m venv venv
source venv/bin/activate
# or source venv/Scripts/activate on windows
pip3 install -r requirements.txt
```

---

## Current Tools

I have so many hodge podge scripts I am going to be moving over to this that I use in my ML work. But this is what is
here so far.

---

### LoRA (lierla), LoCON (LyCORIS) extractor

It is based on the extractor in the [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) tool, but adding some QOL features
and LoRA (lierla) support. It can do multiple types of extractions in one run.
It all runs off a config file, which you can find an example of in  `config/examples/extract.example.yml`.
Just copy that file, into the `config` folder, and rename it to `whatever_you_want.yml`.
Then you can edit the file to your liking. and call it like so:

```bash
python3 run.py config/whatever_you_want.yml
```

You can also put a full path to a config file, if you want to keep it somewhere else.

```bash
python3 run.py "/home/user/whatever_you_want.yml"
```

More notes on how it works are available in the example config file itself. LoRA and LoCON both support
extractions of 'fixed', 'threshold', 'ratio', 'quantile'. I'll update what these do and mean later.
Most people used fixed, which is traditional fixed dimension extraction.

`process` is an array of different processes to run. You can add a few and mix and match. One LoRA, one LyCON, etc.

---

### LoRA Rescale

Change `<lora:my_lora:4.6>` to `<lora:my_lora:1.0>` or whatever you want with the same effect. 
A tool for rescaling a LoRA's weights. Should would with LoCON as well, but I have not tested it.
It all runs off a config file, which you can find an example of in  `config/examples/mod_lora_scale.yml`.
Just copy that file, into the `config` folder, and rename it to `whatever_you_want.yml`.
Then you can edit the file to your liking. and call it like so:

```bash
python3 run.py config/whatever_you_want.yml
```

You can also put a full path to a config file, if you want to keep it somewhere else.

```bash
python3 run.py "/home/user/whatever_you_want.yml"
```

More notes on how it works are available in the example config file itself. This is useful when making 
all LoRAs, as the ideal weight is rarely 1.0, but now you can fix that. For sliders, they can have weird scales form -2 to 2
or even -15 to 15. This will allow you to dile it in so they all have your desired scale

---

### LoRA Slider Trainer

This is how I train most of the recent sliders I have on Civitai, you can check them out in my [Civitai profile](https://civitai.com/user/Ostris/models).
It is based off the work by [p1atdev/LECO](https://github.com/p1atdev/LECO) and [rohitgandikota/erasing](https://github.com/rohitgandikota/erasing)
But has been heavily modified to create sliders rather than erasing concepts. I have a lot more plans on this, but it is
very functional as is. It is also very easy to use. Just copy the example config file in `config/examples/train_slider.example.yml`
to the `config` folder and rename it to `whatever_you_want.yml`. Then you can edit the file to your liking. and call it like so:

```bash
python3 run.py config/whatever_you_want.yml
```

There is a lot more information in that example file. You can even run the example as is without any modifications to see
how it works. It will create a slider that turns all animals into dogs(neg) or cats(pos). Just run it like so:

```bash
python3 run.py config/examples/train_slider.example.yml
```

And you will be able to see how it works without configuring anything. No datasets are required for this method.
I will post an better tutorial soon. 

---

## WIP Tools


### VAE (Variational Auto Encoder) Trainer

This works, but is not ready for others to use and therefore does not have an example config. 
I am still working on it. I will update this when it is ready.
I am adding a lot of features for criteria that I have used in my image enlargement work. A Critic (discriminator),
content loss, style loss, and a few more. If you don't know, the VAE
for stable diffusion (yes even the MSE one, and SDXL), are horrible at smaller faces and it holds SD back. I will fix this.
I'll post more about this later with better examples later, but here is a quick test of a run through with various VAEs.
Just went in and out. It is much worse on smaller faces than shown here.

<img src="https://raw.githubusercontent.com/ostris/ai-toolkit/main/assets/VAE_test1.jpg" width="768" height="auto"> 

---

## TODO
- [X] Add proper regs on sliders
- [X] Add SDXL support (base model only for now)
- [ ] Add plain erasing
- [ ] Make Textual inversion network trainer (network that spits out TI embeddings)

---

## Change Log

#### 2021-08-01
Major changes and update. New LoRA rescale tool, look above for details. Added better metadata so
Automatic1111 knows what the base model is. Added some experiments and a ton of updates. This thing is still unstable
at the moment, so hopefully there are not breaking changes. 

Unfortunately, I am too lazy to write a proper changelog with all the changes.

I added SDXL training to sliders... but.. it does not work properly. 
The slider training relies on a model's ability to understand that an unconditional (negative prompt)
means you do not want that concept in the output. SDXL does not understand this for whatever reason, 
which makes separating out
concepts within the model hard. I am sure the community will find a way to fix this 
over time, but for now, it is not 
going to work properly. And if any of you are thinking "Could we maybe fix it by adding 1 or 2 more text
encoders to the model as well as a few more entirely separate diffusion networks?" No. God no. It just needs a little
training without every experimental new paper added to it. The KISS principal. 


#### 2021-07-30
Added "anchors" to the slider trainer. This allows you to set a prompt that will be used as a 
regularizer. You can set the network multiplier to force spread consistency at high weights

