# SheetSage2-Refined (vendored)

This folder is an unmodified copy of the `sheetsage2_refined` package from
[anime-song/SheetSage2-Refined](https://github.com/anime-song/SheetSage2-Refined) at commit
`edb9e7562be134aac92d1eadf5bfe868453d9386` (0.5.2), with its `LICENSE`.

- **Code:** MIT, Copyright (c) 2026 anime-song (see [LICENSE](LICENSE)).
- **Weights:** not included. They are CC BY-NC 4.0 and come from
  [anime-song/SheetSage2-Refined on Hugging Face](https://huggingface.co/anime-song/SheetSage2-Refined) at revision
  `5e9c494941b8c0bb322d9f17f365a6f886d5e562` (0.5.2), downloaded into the models folder on first use. The encoder they sit
  on is SheetSage2's, read from the same `sheetsage2_bf16.safetensors` the SheetSage2 transcriber uses.

Thanks to anime-song for SheetSage2-Refined: a fine-tuned SheetSage2 with its own decoder and frame heads for beats,
meter, chords and keys, which writes noticeably better bars and timing than the original on real songs.

YuE2 uses it through `src/tokenizer.py` (`RefinedTranscriber`); nothing here is imported unless
`model_kwargs.sheet_transcriber` is `refined`.
