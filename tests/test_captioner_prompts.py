def test_ideogram4_prompt_imports():
    from extensions_built_in.captioner.prompts.ideogram4_prompt import (
        ideogram4_prompt,
    )

    assert ideogram4_prompt.startswith("\n[META]")
