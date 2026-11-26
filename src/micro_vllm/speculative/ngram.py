from typing import Sequence

from micro_vllm.speculative.drafter import DraftOutput

class NGramDrafter:
    def __init__(self, ngram_size: int = 3):
        if ngram_size <= 0:
            raise ValueError("ngram_size must be a positive integer")
        self.ngram_size = ngram_size

    def propose(self, token_ids: Sequence[int], max_tokens: int) -> DraftOutput:
        if max_tokens <= 0:
            return DraftOutput(tokens_ids=())
        # Convert input_ids to a tuple for caching
        input_tuple = tuple(token_ids)
        if len(input_tuple) < self.ngram_size:
            return DraftOutput(tokens_ids=())

        # Generate n-grams from the input_ids
        start = len(input_tuple) - self.ngram_size
        latest = input_tuple[start:start + self.ngram_size]
        for i in range(len(input_tuple) - self.ngram_size - 1, -1, -1):
            ngram = input_tuple[i:i + self.ngram_size]
            if ngram == latest:
                # Found a matching n-gram, propose the next token
                seg_start = i + self.ngram_size
                if seg_start < len(input_tuple):
                    ngrams = input_tuple[seg_start:seg_start + max_tokens]
                    return DraftOutput(tokens_ids=tuple(ngrams))
                else:
                    return DraftOutput(tokens_ids=())

        return DraftOutput(tokens_ids=())