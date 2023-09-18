import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, random_split

from .utils import causal_mask, get_or_build_tokenizer


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = 0
        # We will only add <s> to input, and </s> to the label
        dec_num_padding_tokens = 0

        # Make sure the number of padding tokens is not negative, if it is the input is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise Exception("Input is too long")

        encoder_input = torch.cat(
            (
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ),
            dim=0,
        )

        # Decoder input: Add only <s> token
        decoder_input = torch.cat(
            (
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ),
            dim=0,
        )

        # Label: Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            # "encoder_mask": (encoder_input != self.pad_token)
            # .unsqueeze(0)
            # .unsqueeze(0)
            # .int(),  # (1, 1, seq_len)
            # "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            # & causal_mask(decoder_input.size(0)).int(),  # (1, seq_len, seq_len),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


class BilingualDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        num_workers=2,
    ):
        super().__init__()
        self.config = config
        self.num_workers = num_workers
        self.batch_size = config["batch_size"]

    def prepare_data(self):
        load_dataset(
            "opus_books",
            f'{self.config["lang_src"]}-{self.config["lang_tgt"]}',
            split="train",
        )

    def setup(self, stage=None):
        self.ds_raw = load_dataset(
            "opus_books",
            f'{self.config["lang_src"]}-{self.config["lang_tgt"]}',
            split="train",
        )

        # Build tokenizers
        self.tokenizer_src = get_or_build_tokenizer(
            self.config, self.ds_raw, self.config["lang_src"]
        )
        self.tokenizer_tgt = get_or_build_tokenizer(
            self.config, self.ds_raw, self.config["lang_tgt"]
        )

        # Sort the train_dataset by length of sentences
        self.ds_raw = sorted(
            self.ds_raw,
            key=lambda x: len(x["translation"][self.config["lang_src"]]),
        )

        # Remove any sentences that are longer than 150 tokens from both src and tgt, from the train dataset list
        self.ds_raw = [
            k
            for k in self.ds_raw
            if len(k["translation"][self.config["lang_src"]]) <= 150
            and len(k["translation"][self.config["lang_tgt"]]) <= 150
        ]

        # Remove any sentences where length of src + 10 < length of tgt
        self.ds_raw = [
            k
            for k in self.ds_raw
            if len(k["translation"][self.config["lang_src"]]) + 10
            >= len(k["translation"][self.config["lang_tgt"]])
        ]

        # Keep 90% of the data for training, 10% for validation
        train_ds_size = int(0.9 * len(self.ds_raw))
        val_ds_size = len(self.ds_raw) - train_ds_size
        self.train_ds_raw, self.val_ds_raw = random_split(
            self.ds_raw, [train_ds_size, val_ds_size]
        )

        self.train_ds = BilingualDataset(
            self.train_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["lang_src"],
            self.config["lang_tgt"],
            self.config["seq_len"],
        )
        self.val_ds = BilingualDataset(
            self.val_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["lang_src"],
            self.config["lang_tgt"],
            self.config["seq_len"],
        )

        # Find the max length of the sentences in source and target language
        max_len_src = 0
        max_len_tgt = 0

        for item in self.train_ds_raw:
            max_len_src = max(
                len(
                    self.tokenizer_src.encode(
                        item["translation"][self.config["lang_src"]]
                    ).ids
                ),
                max_len_src,
            )
            max_len_tgt = max(
                len(
                    self.tokenizer_tgt.encode(
                        item["translation"][self.config["lang_tgt"]]
                    ).ids
                ),
                max_len_tgt,
            )

        print("Max length of source sentences:", max_len_src)
        print("Max length of target sentences:", max_len_tgt)

    def collate_fn(self, batch):
        return dynamic_padding_collate_fn(batch, self.tokenizer_tgt)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


def dynamic_padding_collate_fn(batch, tokenizer):
    max_encoder_input_len = max([len(item["encoder_input"]) for item in batch])
    max_decoder_input_len = max([len(item["decoder_input"]) for item in batch])

    encoder_inputs = []
    decoder_inputs = []
    labels = []
    for item in batch:
        encoder_input = item["encoder_input"]
        decoder_input = item["decoder_input"]
        label = item["label"]

        # Pad the encoder input
        encoder_input = torch.cat(
            (
                encoder_input,
                torch.tensor(
                    [tokenizer.token_to_id("[PAD]")]
                    * (max_encoder_input_len - len(encoder_input)),
                    dtype=torch.int64,
                ),
            ),
            dim=0,
        )

        # Pad the decoder input
        decoder_input = torch.cat(
            (
                decoder_input,
                torch.tensor(
                    [tokenizer.token_to_id("[PAD]")]
                    * (max_decoder_input_len - len(decoder_input)),
                    dtype=torch.int64,
                ),
            ),
            dim=0,
        )

        # Pad the label
        label = torch.cat(
            (
                label,
                torch.tensor(
                    [tokenizer.token_to_id("[PAD]")]
                    * (max_decoder_input_len - len(label)),
                    dtype=torch.int64,
                ),
            ),
            dim=0,
        )

        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        labels.append(label)

    encoder_inputs = torch.stack(encoder_inputs)
    decoder_inputs = torch.stack(decoder_inputs)
    encoder_mask = (
        (encoder_inputs != tokenizer.token_to_id("[PAD]"))
        .unsqueeze(1)
        .unsqueeze(1)
        .int()
    )
    decoder_mask = (
        causal_mask(decoder_inputs.size(1))
        .repeat(decoder_inputs.size(0), 1, 1, 1)
        .int()
    ) & (
        (decoder_inputs != tokenizer.token_to_id("[PAD]"))
        .unsqueeze(1)
        .unsqueeze(1)
        .int()
    )

    labels = torch.stack(labels)
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]

    return {
        "encoder_input": encoder_inputs,
        "decoder_input": decoder_inputs,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask,
        "label": labels,
        "src_text": src_texts,
        "tgt_text": tgt_texts,
    }
