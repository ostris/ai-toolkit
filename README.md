# AI Toolkit by Ostris

WIP for now, but will be a collection of tools for AI tools as I need them.

## Installation

I will try to update this to be more beginner-friendly, but for now I am assuming
a general understanding of python, pip, pytorch, and using virtual environments:

Linux:

```bash
pythion3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
pythion3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Current Tools

### LyCORIS extractor

It is similar to the [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) tool, but adding some QOL features.
It all runs off a config file, which you can find an example of in  `config/examples/locon_config.example.json`.
Just copy that file, into the `config` folder, and rename it to `whatever_you_want.json`.
Then you can edit the file to your liking. and call it like so:

```bash
python3 scripts/extract_locon.py "whatever_you_want"
```

You can also put a full path to a config file, if you want to keep it somewhere else.

```bash
python3 scripts/extract_locon.py "/home/user/whatever_you_want.json"
```

File name is auto generated and dumped into the `output` folder. You can put whatever meta you want in the
`meta` section of the config file, and it will be added to the metadata of the output file. I just have
some recommended fields in the example file. The script will add some other useful metadata as well.

process is an array or different processes to run on the conversion to test. You will normally just need one though.

Will update this later.

