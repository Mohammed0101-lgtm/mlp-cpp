import torch
from torch import nn, Tensor


class Config(object):
    def __init__(self, _embedding_dim: int, _vocab_size: int, _hidden_layer_size: int, _output_layer_size: int):
        self.embd_dims = _embedding_dim
        self.vocab_size = _vocab_size
        self.hidden_layer_size = _hidden_layer_size
        self.output_layer_size = _output_layer_size


class MLP(nn.Module):
    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embd = nn.Embedding(config.vocab_size, config.embd_dims)
        self.hidden_layer = nn.Linear(
            config.hidden_layer_size, config.output_layer_size)
        self.output_layer = nn.Linear(config.output_layer_size, 1)

    def forward(input_data: Tensor) -> Tensor:
        pass

    def backward() -> None:
        pass


class DataLoader(object):
    def __init__(self):
        pass

    def encode(self, input_str: str) -> Tensor:
        if not input_str:
            raise ValueError("Input string cannot be empty")

        encoded_indices = [ord(c) % self.vocab_size for c in input_str]
        encoded = torch.tensor(encoded_indices, dtype=torch.long)
        return encoded

    def decode(self, input_tokens: Tensor) -> str:
        if input_tokens.numel() == 0:
            raise ValueError("Input tokens cannot be empty")

        decoded = ''.join(chr(token.item()) for token in input_tokens)
        return decoded
    
    def read_file(filename: str) -> str:
        assert filename
        with open(filename) as f:
            text = filename.read()
        
        return text

    
