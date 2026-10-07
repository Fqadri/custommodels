import tiktoken


# GPT-2 uses byte-level Byte Pair Encoding (BPE), implemented here by tiktoken.
class GPT2Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"}) # this is EOS special token, GPT 2 specific, which indicates the end of the text sequence.

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def eot_token(self) -> int:
        return self.tokenizer.eot_token
