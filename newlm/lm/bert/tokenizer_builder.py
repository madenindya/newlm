from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from ...utils.file_util import create_dir


class TokenizerBuilder:
    def __init__(
        self,
        configs,
    ):
        self.configs = configs

    def create(self, input_dir: str, output_dir: str) -> BertWordPieceTokenizer:
        paths = [str(x) for x in Path(input_dir).glob("**/*.txt")]
        paths = list(filter(lambda x: "cache" not in x, paths))
        create_dir(output_dir)
        tokenizer = BertWordPieceTokenizer()
        tokenizer.train(
            files=paths,
            **self.configs,
        )
        tokenizer.save_model(output_dir)
        return tokenizer
