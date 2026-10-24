from pathlib import Path

import sentencepiece as spm


class LlamaTokenizer:
    def __init__(self, tokenizer_file: str | Path):
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer_file))
        self.tokenizer = sp
        if self.bos_id < 0 or self.eos_id < 0:
            raise ValueError("The LLaMA tokenizer must define BOS and EOS tokens.")

    def encode(self, text: str) -> list[int]:
        # LLaMA prompts start with BOS; EOS is left for the model to generate.
        ids = self.tokenizer.encode(text, out_type=int)
        return [self.bos_id, *ids]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def bos_id(self) -> int:
        return self.tokenizer.bos_id()

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_piece_size()