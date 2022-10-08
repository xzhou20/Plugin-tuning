import argparse
import logging
import math
import os
import random
import time
import numpy as np
import datasets
import torch
import loralib as lora
from torch.optim import Adam
from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from  torch.optim.lr_scheduler import CosineAnnealingLR

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM, 
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.dummy_pt_objects import BertForMaskedLM
# from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='conll',
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(        
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default='./dataset/ner/conll/train.json', help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--test_file", type=str, default='./dataset/ner/conll/test.json', help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default='./dataset/ner/conll/dev.json', help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default='roberta-base'
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk", 'ae', 'slot'],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--label_schema",
        default='BIO',
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--target_metric",
        default='f1',
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--prefix_len",
        default=5,
        type=int,
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--prefix_dropout",
        default=0.1,
        type=float,
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--mask_prob",
        default=0.1,
        type=float,
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--mid_dim",
        default=300,
        type=int,
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--start_eval_epoch",
        default=0,
        type=int,
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--add_special_token",
        default=0,
        type=int,
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--update_emb",
        default=0,
        type=int,
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--save_model",
        default=0,
        type=int,
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--rank",
        default=8,
        type=int,
        help="rank for LoRA",
    )
    parser.add_argument(
        "--lora_alpha",
        default=8,
        type=int,
        help="rank for LoRA",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
        help="rank for LoRA",
    )
    parser.add_argument(
        "--log_path",
        default='/root/plug_lora/log/roberta-base.txt',
        type=str,
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--prefix_length", type=int, default=5)
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        seed_everything(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    # else:
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    extension = args.train_file.split(".")[-1]

    raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if args.label_column_name is not None:
        label_column_name = args.label_column_name
    elif f"{args.task_name}_tags" in column_names:
        label_column_name = f"{args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    import json
    independent_task = ['chunk', 'ner']
    task_map = {'ner':
                        {
                        'conll': 
                                {'B-LOC': 'Germany', 'B-PER': 'Michael', 'I-PER': 'van', 'I-LOC': 'Louis', 'B-MISC': 'German', 'I-MISC': 'Series', 'B-ORG': 'Inc', 'I-ORG': 'News'},
                                # {'B-LOC': 'Australia', 'B-PER': 'David', 'I-PER': 'van', 'I-LOC': 'Louis', 'B-MISC': 'Pol', 'I-MISC': 'Series', 'B-ORG': 'Reuters', 'I-ORG': 'News'},
                                # {'B-ORG': 'Reuters', 'B-MISC': 'Israeli', 'B-PER': 'Mark', 'I-PER': 'Mal', 'B-LOC': 'Britain', 'I-ORG': 'News', 'I-MISC': 'DIV', 'I-LOC': 'Arab'}, # train frac
                                # {"B-ORG": "Reuters", "B-MISC": "Israeli", "B-PER": "Mark", "I-PER": "Mark", "B-LOC": "Britain", "I-ORG": "Reuters", "I-MISC": "Israeli", "I-LOC": "Britain"},
                                # {"B-PER":"John", "I-PER": "John","B-ORG":'Corporation', "I-ORG": "Corporation", "B-LOC":"Australia", "I-LOC": "Australia", "B-MISC":"American", "I-MISC": "American"},
                        'ontonotes':
                                # {'I-CARDINAL': 'thousands', 'I-DATE': 'years', 'I-EVENT': 'War', 'I-FAC': 'Airport', 'I-GPE': 'China', 'I-LANGUAGE': 'English', 'I-LAW': 'Act', 'I-LOC': 'Middle', 'I-MONEY': 'billion', 'I-NORP': 'Chinese', 'I-ORDINAL': 'seventh', 'I-ORG': 'National', 'I-PERCENT': 'percent', 'I-PERSON': 'Bush', 'I-PRODUCT': 'USS', 'I-QUANTITY': 'kilometers', 'I-TIME': 'minutes', 'I-WORK_OF_ART': 'Prize', 'B-CARDINAL': 'thousands', 'B-DATE': 'years', 'B-EVENT': 'War', 'B-FAC': 'Airport', 'B-GPE': 'China', 'B-LANGUAGE': 'English', 'B-LAW': 'Act', 'B-LOC': 'Middle', 'B-MONEY': 'billion', 'B-NORP': 'Chinese', 'B-ORDINAL': 'seventh', 'B-ORG': 'National', 'B-PERCENT': 'percent', 'B-PERSON': 'Bush', 'B-PRODUCT': 'USS', 'B-QUANTITY': 'kilometers', 'B-TIME': 'minutes', 'B-WORK_OF_ART': 'Prize'}
                                # {'I-WORK_OF_ART': 'Life', 'I-CARDINAL': 'three', 'I-LANGUAGE': 'language', 'I-QUANTITY': 'feet', 'I-ORDINAL': 'second', 'I-PERCENT': 'percent', 'I-PRODUCT': 'ship', 'I-PERSON': 'John', 'I-EVENT': 'Christmas', 'I-MONEY': 'millions', 'I-DATE': 'years', 'I-NORP': 'Arab', 'I-TIME': 'evening', 'I-FAC': 'Center', 'I-GPE': 'China', 'I-LAW': 'Laws', 'I-LOC': 'South', 'I-ORG': 'Corporation', 'B-WORK_OF_ART': 'Life', 'B-CARDINAL': 'three', 'B-LANGUAGE': 'language', 'B-QUANTITY': 'feet', 'B-ORDINAL': 'second', 'B-PERCENT': 'percent', 'B-PRODUCT': 'ship', 'B-PERSON': 'John', 'B-EVENT': 'Christmas', 'B-MONEY': 'millions', 'B-DATE': 'years', 'B-NORP': 'Arab', 'B-TIME': 'evening', 'B-FAC': 'Center', 'B-GPE': 'China', 'B-LAW': 'Laws', 'B-LOC': 'South', 'B-ORG': 'Corporation'} # 小马博士
                                # {'I-WORK_OF_ART': 'Life', 'I-CARDINAL': 'three', 'I-LANGUAGE': 'English', 'I-QUANTITY': 'feet', 'I-ORDINAL': 'second', 'I-PERCENT': 'percent', 'I-PRODUCT': 'USS', 'I-PERSON': 'Bush', 'I-EVENT': 'War', 'I-MONEY': 'billion', 'I-DATE': 'years', 'I-NORP': 'Chinese', 'I-TIME': 'PM', 'I-FAC': 'Airport', 'I-GPE': 'China', 'I-LAW': 'Act', 'I-LOC': 'Middle', 'I-ORG': 'National', 'B-WORK_OF_ART': 'Life', 'B-CARDINAL': 'three', 'B-LANGUAGE': 'English', 'B-QUANTITY': 'feet', 'B-ORDINAL': 'second', 'B-PERCENT': 'percent', 'B-PRODUCT': 'USS', 'B-PERSON': 'Bush', 'B-EVENT': 'War', 'B-MONEY': 'billion', 'B-DATE': 'years', 'B-NORP': 'Chinese', 'B-TIME': 'PM', 'B-FAC': 'Airport', 'B-GPE': 'China', 'B-LAW': 'Act', 'B-LOC': 'Middle', 'B-ORG': 'National'} # mix up
                                # {'B-CARDINAL': 'thirds', 'I-CARDINAL': 'thirds', 'B-PRODUCT': 'USS', 'I-PRODUCT': 'USS', 'B-DATE': '1988', 'I-DATE': '1988', 'B-LOC': 'Gal', 'I-LOC': 'Gal', 'B-NORP': 'Democrats', 'I-NORP': 'Democrats', 'B-GPE': 'Britain', 'I-GPE': 'Britain', 'B-PERSON': 'Bar', 'I-PERSON': 'Bar', 'B-TIME': 'PM', 'I-TIME': 'PM', 'B-ORG': 'Corp', 'I-ORG': 'Corp', 'B-WORK_OF_ART': 'Hard', 'I-WORK_OF_ART': 'Hard', 'B-QUANTITY': 'met', 'I-QUANTITY': 'met', 'B-PERCENT': '11', 'I-PERCENT': '11', 'B-EVENT': 'Water', 'I-EVENT': 'Water', 'B-MONEY': 'fr', 'I-MONEY': 'fr', 'B-ORDINAL': 'se', 'I-ORDINAL': 'se', 'B-FAC': 'Square', 'I-FAC': 'Square', 'B-LAW': 'Act', 'I-LAW': 'Act', 'B-LANGUAGE': 'Mand', 'I-LANGUAGE': 'Mand'} # 频率
                                {'B-CARDINAL': 'do', 'I-CARDINAL': 'do', 'B-ORDINAL': 'fifth', 'I-ORDINAL': 'fifth', 'B-PERSON': 'Bush', 'I-PERSON': 'Bush', 'B-ORG': 'Inc', 'I-ORG': 'Inc', 'B-GPE': 'British', 'I-GPE': 'Israel', 'B-WORK_OF_ART': 'Rel', 'I-WORK_OF_ART': 'Rel', 'B-DATE': 'September', 'I-DATE': 'September', 'B-TIME': 'noon', 'I-TIME': 'noon', 'B-NORP': 'Palest', 'I-NORP': 'Palest', 'B-LAW': 'Am', 'I-LAW': 'Am', 'B-LOC': 'Sea', 'I-LOC': 'Sea', 'B-QUANTITY': 'kil', 'I-QUANTITY': 'kil', 'B-EVENT': 'Games', 'I-EVENT': 'Games', 'B-MONEY': 'US', 'I-MONEY': 'US', 'B-FAC': 'Ring', 'I-FAC': 'Ring', 'B-PERCENT': 'percent', 'I-PERCENT': 'percent', 'B-PRODUCT': 'Cole', 'I-PRODUCT': 'Cole', 'B-LANGUAGE': 'English', 'I-LANGUAGE': 'English'}, # frac from test
                        'ace2005':
                            # {'B-GPE': 'country', 'B-PER': 'she', 'I-PER': 'hus', 'B-VEH': 'plane', 'I-VEH': 'odore', 'B-ORG': 'military', 'B-FAC': 'store', 'B-LOC': 'border', 'I-GPE': 'Franc', 'I-LOC': 'olan', 'I-ORG': 'Qaida', 'I-FAC': 'international', 'B-WEA': 'weapons', 'I-WEA': 'mil'},
                            {'B-VEH': 'plane', 'B-GPE': 'British', 'I-GPE': 'states', 'B-FAC': 'complex', 'I-FAC': 'international', 'B-PER': 'you', 'I-VEH': 'odore', 'B-ORG': 'military', 'I-ORG': 'Party', 'I-PER': 'Mas', 'B-LOC': 'West', 'B-WEA': 'weapons', 'I-LOC': 'Bank', 'I-WEA': 'mil'},
                            # {'B-VEH': 'plane', 'B-GPE': 'states', 'I-GPE': 'states', 'B-FAC': 'complex', 'I-FAC': 'complex', 'B-PER': 'you', 'I-VEH': 'plane', 'B-ORG': 'military', 'I-ORG': 'military', 'I-PER': 'you', 'B-LOC': 'West', 'B-WEA': 'weapons', 'I-LOC': 'West', 'I-WEA': 'weapons'},
                        'WNUT17':{'B-location': 'San', 'I-location': 'ville', 'B-group': 'Liverpool', 'B-corporation': 'YouTube', 'B-person': 'Pope', 'B-creative-work': 'Iron', 'B-product': 'iPhone', 'I-person': 'Jonathan', 'I-creative-work': 'Kill', 'I-corporation': 'uc', 'I-group': 'ista', 'I-product': 'Jump'}
                        },
                        
                        # {"B-ORG": "Reuters", "B-MISC": "Israeli", "B-PER": "Mark", "I-PER": "Mark", "B-LOC": "Britain", "I-ORG": "Reuters", "I-MISC": "Israeli", "I-LOC": "Britain"},
                        # {"B-ORG": "Corporation", "B-MISC": "American", "B-PER": "John", "I-PER": "John", "B-LOC": "Australia", "I-ORG": "Corporation", "I-MISC": "American", "I-LOC": "Australia"},
                        # {"B-ORG": ["Corporation", "Reuters"], "B-MISC": ["American", "Israeli"], "B-PER": ["John", "Mark"], "I-PER": ["John", "Mark"], "B-LOC": ["Australia","Britain"], "I-ORG": ["Corporation", "Reuters"], "I-MISC": ["American", "Israeli"], "I-LOC": ["Australia","Britain"]},
                'chunk':
                        {
                            'conll':{'B-NP': 'it', 'B-VP': 'were', 'I-NP': 'percent', 'B-PP': 'from', 'B-SBAR': 'while', 'B-ADVP': 'when', 'I-VP': 'res', 'B-ADJP': 'ready', 'B-PRT': 'UP', 'B-LST': '(', 'I-LST':'212', 'I-ADJP': 'trans', 'B-CONJP': 'rather', 'I-CONJP': 'Not', 'I-PP': 'Near', 'B-INTJ': 'BE', 'I-ADVP': 'strong', 'I-SBAR': 'whether', 'I-INTJ': 'makeupword1', 'I-PRT':'makeupword0'},
                            'conll2000':{"B-NP": "it", "B-PP": "against", "I-NP": "million", "B-VP": "were", "I-VP": "produ", "B-SBAR": "If", "B-ADJP": "able", "B-ADVP": "When", "I-ADVP": "forth", "I-ADJP": "un", "I-SBAR": "if", "I-PP": "as", "B-PRT": "UP", "B-LST": "1", "B-INTJ": "Please", "I-INTJ": "BE", "B-CONJP": "rather", "I-CONJP": "than", "I-PRT": "makeupword0", "B-UCP": "wine", "I-UCP": "makeupword1", "I-LST":"makeupword2"}
                            # 'conll':{'B-NP': 'The', 'I-NP': 'The', 'B-VP': 'said', 'I-VP': 'said', 'B-PP': 'into', 'I-PP': 'into', 'B-SBAR': 'whether', 'I-SBAR': 'whether', 'B-ADJP': 'able', 'I-ADJP': 'able', 'B-ADVP': 'where', 'I-ADVP': 'where', 'B-PRT': 'UP', 'I-PRT': 'UP', 'B-CONJP': 'Not', 'I-CONJP': 'Not', 'B-INTJ': 'BE', 'I-INTJ': 'BE', 'B-LST': '(', 'I-LST': '('}
                            # 'conll':{'B-NP': 'who', 'B-VP': 'has', 'I-NP': 'country', 'I-VP': 'done', 'B-PP': 'into', 'B-SBAR': 'whether', 'B-ADJP': 'able', 'I-ADJP': 'resistant', 'B-ADVP': 'when', 'B-PRT': 'UP', 'B-CONJP': 'Not', 'I-CONJP': 'NONE', 'I-PP': 'Near', 'B-INTJ': 'O', 'I-ADVP': 'madeupword0001', 'B-LST': '(', 'I-SBAR': 'madeupword0002', 'I-LST': 'NONE', 'I-INTJ': 'O'},
                        },
                'pos':
                    {
                        'wsj':{"NNP": "Mr", ",": ",", "CD": "million", "NNS": "years", "JJ": "new", "MD": "would", "VB": "dis", "DT": "those", "NN": "company", "IN": "from", ".": ".", "VBZ": "does", "VBG": "according", "CC": "or", "VBD": "was", "VBN": "been", "RB": "also", "TO": "To", "PRP": "it", "RBR": "sm", "WDT": "whatever", "VBP": "'re", "RP": "UP", "PRP$": "its", "JJS": "largest", "POS": "'", "''":"''", "``": "``", "EX": "EX", "\"\"": "\"\"", "WP": "who", ":": "--", "JJR": "small", "WRB": "when", "$": "$", "NNPS": "Tex", "WP$": "whose", "-LRB-": "-LRB-", "-RRB-": "-RRB-", "PDT": "all", "RBS": "most", "FW": "bon", "UH": "Oh", "SYM": "FF", "LS": "LS", "#": "#"},
                        'conll':{'NNP': 'Thursday', 'VBZ': 'is', 'JJ': 'new', 'NN': 'year', 'TO': 'to', 'VB': 'be', '.': '.', 'CD': 'two', 'DT': 'the', 'VBD': 'was', 'IN': 'of', 'PRP': 'he', 'NNS': 'years', 'VBP': 'are', 'MD': 'would', 'VBN': 'seen', 'POS': 'Women', 'JJR': 'we', '"': '"', 'RB': 'not', ',': ',', 'FW': 'der', 'CC': 'and', 'WDT': 'which', '(': '(', ')': ')', ':': ':', 'PRP$': 'his', 'RBR': 'Earlier', 'VBG': 'being', 'EX': 'there', 'WP': 'who', 'WRB': 'when', '$': 'US', 'RP': 'UP', 'NNPS': 'Palest', 'SYM': '------------------------', 'RBS': 'most', 'UH': 'Def', 'PDT': 'Quite', "''": "''", 'LS': '2002', 'JJS': 'le', 'WP$': 'whose', 'NN|SYM': 'Jul'}
                        # 'conll':{'NN': 'government', ':': '--', 'NNP': 'Friday', 'VB': 'be', ',': 'NONE', 'IN': 'of', 'DT': 'the', '.': 'NONE', 'NNPS': 'It', 'CD': '1996', 'VBD': 'was', 'PRP$': 'his', 'JJ': 'other', 'VBP': 'are', 'CC': 'and', 'PRP': 'he', 'VBG': 'being', 'TO': 'to', 'NNS': 'people', 'JJS': 'le', 'WRB': 'when', 'RB': 'not', 'WDT': 'which', 'VBN': 'been', 'POS': 'NONE', 'RP': 'UP', 'VBZ': 'is', "''": 'NONE', 'JJR': 'strong', 'WP': 'who', '"': 'NONE', 'MD': 'would', '(': 'NONE', ')': 'NONE', 'LS': 'NONE', 'SYM': 'la', '$': 'US', 'FW': 'est', 'RBS': 'NONE', 'EX': 'There', 'RBR': 'Earlier', 'WP$': 'whose', 'PDT': 'NONE', 'UH': 'yes'},
                        # 'conll':{'NNP': 'Thursday', 'VBZ': 'is', 'JJ': 'new', 'NN': 'year', 'TO': 'to', 'VB': 'be', '.': 'NONE', 'CD': 'two', 'DT': 'the', 'VBD': 'was', 'IN': 'of', 'PRP': 'he', 'NNS': 'years', 'VBP': 'are', 'MD': 'would', 'VBN': 'been', 'POS': 'NONE', 'JJR': 'we', '"': 'NONE', 'RB': 'not', ',': 'NONE', 'FW': 'der', 'CC': 'and', 'WDT': 'which', '(': 'NONE', ')': 'NONE', ':': '--', 'PRP$': 'his', 'RBR': 'Earlier', 'VBG': 'being', 'EX': 'NONE', 'WP': 'who', 'WRB': 'when', '$': 'US', 'RP': 'UP', 'NNPS': 'Palest', 'SYM': '------------------------', 'RBS': 'NONE', 'UH': 'Def', 'PDT': 'Quite', "''": 'NONE', 'LS': 'NONE', 'JJS': 'le', 'WP$': 'whose', 'NN|SYM': 'NONE'}
                    },
                'ae':
                    {
                        'laptop':{'B-aspect':'Windows', 'I-aspect':'card'}
                    }
                    # {'NNP': 'Thursday', 'VBZ': 'is', 'JJ': 'new', 'NN': 'year', 'TO': 'to', 'VB': 'be', '.': '.', 'CD': '0', 'DT': 'the', 'VBD': 'was', 'IN': 'of', 'PRP': 'he', 'NNS': 'years', 'VBP': 'are', 'MD': 'would', 'VBN': 'been', 'POS': 'Women', 'JJR': 'Feb', '"': '"', 'RB': 'not', ',': ',', 'FW': 'der', 'CC': 'and', 'WDT': 'which', '(': '(', ')': ')', ':': ':', 'PRP$': 'his', 'RBR': 'Earlier', 'VBG': 'being', 'EX': 'there', 'WP': 'who', 'WRB': 'when', '$': '$', 'RP': 'UP', 'NNPS': 'Games', 'SYM': '/', 'RBS': 'most', 'UH': 'O', 'PDT': 'Quite', "''": "''", 'LS': '2002', 'JJS': 'latest', 'WP$': 'whose', 'NN|SYM': 'NONE'}
                    # {"NNP": "Friday", "VBZ": "have", "JJ": "first", "NN": "year", "TO": "into", "VB": "take", ".": ".", "CD": "three", "DT": "another", "VBD": "were", "IN": "after", "PRP": "they", "NNS": "people", "VBP": "haven", "MD": "would", "VBN": "been", "POS": "Women", "JJR": "more", "\"": "\"", "RB": "also", ",": ",", "FW": "versus", "CC": "then", "WDT": "which", "(": "(", ")": ")", ":": ":", "PRP$": "them", "RBR": "less", "VBG": "being", "EX": "there", "WP": "whom", "WRB": "when", "$": "$", "RP": "down", "NNPS": "Indians", "SYM": "each", "RBS": "most", "UH": "yeah", "PDT": "everything", "''": "''", "LS": "2002", "JJS": "best", "WP$": "whose", "NN|SYM": "television"}
                }
    label_token_map = task_map[args.task_name][args.dataset_name]
    label_list = list(label_token_map.keys())

    label_list += 'O'
    label_to_id = {l: i for i, l in enumerate(label_list)}
    print(label_to_id)
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    print('load model')
    # config.hidden_dropout_prob = 0
    from transformers import AutoModelForMaskedLM
    from transformers.adapters import PrefixTuningConfig
    
    model = AutoModelForMaskedLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config
            )
    
    adapter_config = PrefixTuningConfig(flat=False, prefix_length=args.prefix_length, bottleneck_size=300)
    model.add_adapter("prefix_tuning", config=adapter_config)
    model.set_active_adapters("prefix_tuning")
    model.train_adapter(["prefix_tuning"])
    print('done')
    if 'roberta' in args.model_name_or_path:
        num_tokens, _ = model.roberta.embeddings.word_embeddings.weight.shape
        # tokenizer = add_label_token_roberta(model, tokenizer, label_token_map)
        # label_token_map = {item:item for item in label_token_map}
    else:
        num_tokens, _ = model.bert.embeddings.word_embeddings.weight.shape
    
    # label_token_map = {item:item for item in label_token_map}


    label_token_to_id = {label: tokenizer.convert_tokens_to_ids(label_token) for label, label_token in label_token_map.items()}
    # label_token_id_to_label = {idx:label for label,idx in label_token_to_id.items() if label.startswith('I-')}
    # 把B的id映射到I上面去，在evalute的时候拼接
    label_token_id_to_label = {idx:label  for label,idx in label_token_to_id.items()}
    # 
    if args.task_name not in independent_task:
        label_token_id_to_label = {}
        for label, idx in label_token_to_id.items():
            if label.startswith('B-'):
                label_token_id_to_label[idx] = 'I-'+label[2:]
            else:
                label_token_id_to_label[idx] = label
    print(label_token_map)
    # model._init_prefix_embedding(set(label_token_id_to_label.keys()))
    
   
    # Preprocessing the datasets.
    # First we tokenize all the texts.


    padding = "max_length" if args.pad_to_max_length else False

    def add_special_token_tokenizer(special_token_list, tokenizer, model):
        tokenizer.add_tokens(special_token_list)
        special_token_ids = tokenizer.convert_tokens_to_ids(special_token_list)
        return tokenizer, special_token_ids

    special_token_ids = []
    if args.add_special_token:
        special_token_prefix = '<prefix'
        special_token_list = [special_token_prefix+'{}>'.format(i) for i in range(args.add_special_token)]
        tokenizer, special_token_ids = add_special_token_tokenizer(special_token_list, tokenizer, model)
        model.resize_token_embeddings(len(tokenizer))
        for token_ids in special_token_ids:
            model.roberta.embeddings.word_embeddings.weight.data[token_ids] = model.roberta.embeddings.word_embeddings.weight.data[133]

    def add_special_token(text):
        return special_token_list + text
    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        if args.add_special_token:
            examples[text_column_name] = list(map(add_special_token, examples[text_column_name]))
            examples[label_column_name] = list(map(lambda x:['O']*args.add_special_token+x, examples[label_column_name]))
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        target_tokens = []
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            input_ids = tokenized_inputs.input_ids[i]
            previous_word_idx = None
            label_ids = []
            target_token = []
            for input_idx, word_idx in zip(input_ids, word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None or input_idx in special_token_ids:
                    target_token.append(-100)
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    target_token.append(label_token_to_id[label[word_idx]] if label[word_idx] !='O' else input_idx)
                    # target_tokens.append()
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens else -100)
                    target_token.append(label_token_to_id[label[word_idx]] if label[word_idx] !='O' else input_idx)
                    # target_token.append(input_idx)
                previous_word_idx = word_idx
            target_tokens.append(target_token)
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = target_tokens
        tokenized_inputs['ori_labels'] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
        load_from_cache_file=False
    )

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["test"]
    dev_dataset = processed_raw_datasets["validation"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForLMTokanClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    dev_dataloader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.



    # if args.update_emb:
    #     allow_optimize_list = ['prefix', 'embeddings.word_embeddings']
    # else:
    #     allow_optimize_list = ['prefix']

    allow_optimize_list = ['adapter']
    no_decay = ["bias", "LayerNorm.weight"]
    for name, param in model.named_parameters():
        if any(allow_name in name for allow_name in allow_optimize_list):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(allow_name in n for allow_name in allow_optimize_list)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(allow_name in n for allow_name in allow_optimize_list)],
            "weight_decay": 0.0,
        },
    ]
    # optimizer_grouped_parameters = model.parameters()
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, dev_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, dev_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    args.num_warmup_steps = args.max_train_steps*args.warmup_ratio
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # Metrics
    metric = load_metric("seqeval")
    label_schema = args.label_schema

    def switch_to_BIO(labels):
        past_label = 'O'
        labels_BIO = []
        for label in labels:
            if label.startswith('I-') and (past_label=='O' or past_label[2:]!=label[2:]):
                labels_BIO.append('B-'+label[2:])
            else:
                labels_BIO.append(label)
            past_label = label
        return labels_BIO
    
    def get_predictions_with_subword(y_pred, y_true):
        true_predictions = []
        for pred, gold_label in zip(y_pred, y_true):
            true_pred = []
            # subword_list = []
            for idx, (p, l) in enumerate(zip(pred, gold_label)):
                if l!=-100:
                    # 为上一个词赋label
                    subword_list = [p]
                    subword_idx = idx
                    while(True):
                        subword_idx += 1
                        if subword_idx>=len(gold_label) or gold_label[subword_idx] != -100:
                            break
                        subword_list.append(pred[subword_idx])
                    label_list = [label_token_id_to_label[subword] if subword in label_token_id_to_label.keys() else 'O' for subword in subword_list ]
                    true_pred.append(max(label_list, key=label_list.count))
            true_predictions.append(true_pred)
        return true_predictions

    def get_labels(predictions, references, input_ids=None):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
            if input_ids!=None:
                inputs = input_ids.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()
            if input_ids!=None:
                inputs = input_ids.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_token_id_to_label[p] if p in label_token_id_to_label.keys() else 'O' for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        if label_schema == 'BIO' and args.task_name not in independent_task:
            true_predictions = list(map(switch_to_BIO, true_predictions))
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        
        if args.task_name == 'pos':
            true_predictions = [['B-'+item if item!='O' else 'O' for item in pred_list] for pred_list in true_predictions]
            true_labels = [['B-'+item if item!='O' else 'O' for item in pred_list] for pred_list in true_labels]

        if input_ids!=None:
            inputs =  [
            [i for (i, l) in zip(inp, gold_label) if l != -100]
            for inp, gold_label in zip(inputs, y_true)
            ]
            return true_predictions, true_labels, inputs
        return true_predictions, true_labels

    def compute_metrics():
        results = metric.compute()
        # print(results)
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            print(final_results)
            # return final_results
        # else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def evaluate(target_dataloader):
        eval_batch_loss = 0
        model.eval()
        for step, batch in enumerate(target_dataloader):
            with torch.no_grad():
                ner_label = batch.pop('ori_labels', 'not found ner_labels')
                outputs = model(**batch)
            input_ids = batch['input_ids']
            predictions = outputs.logits.argmax(dim=-1)
            eval_batch_loss += outputs.loss.item()
            labels = ner_label
            
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            preds, refs, input_ids = get_labels(predictions_gathered, labels_gathered, input_ids)
            input_tokens = [tokenizer.convert_ids_to_tokens(item) for item in input_ids]
            metric.add_batch(
                predictions=preds,
                references=refs,
            )  # predictions and preferences are expected to be a nested list of labels, not label_ids
        eval_metric = compute_metrics()
        return eval_metric
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total optimization parameters = {trainable_parameters}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    

    embedding_stop_indices = [i for i in range(num_tokens) if i not in list(label_token_id_to_label.keys())]
    embedding_stop_indices = torch.LongTensor(embedding_stop_indices).to(device)

    completed_steps = 0
    best_result = -1
    best_epoch = 0
    best_test_metric = 0
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            ner_label = batch.pop('ori_labels', 'not found ner_labels')
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            epoch_loss += loss.item()
            accelerator.backward(loss)
            # stopgrad for other embedding 
            if args.update_emb:
                model.roberta.embeddings.word_embeddings.weight.grad[embedding_stop_indices] = 0

            # cal metric 
            predictions = outputs.logits.argmax(dim=-1)
            labels = ner_label
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            preds, refs = get_labels(predictions_gathered, labels_gathered)
            metric.add_batch(
                predictions=preds,
                references=refs,
            )
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_description('loss: %.4f, lr: %f'%(epoch_loss/(step+1), current_lr))
                completed_steps += 1

        train_metric = compute_metrics()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
 
        test_metric = 'None'
        if epoch >=args.start_eval_epoch:
            eval_metric = evaluate(dev_dataloader)
            if eval_metric[args.target_metric]>best_result:
                best_epoch = epoch
                best_result = eval_metric[args.target_metric]
                test_metric = evaluate(eval_dataloader)
                # output_dir = f'./save_models/{args.task_name}/{args.dataset_name}'
                # args.output_dir = output_dir
                # history_result = json.load(open(os.path.join(output_dir, 'best_result.json'), 'r'))
                best_test_metric = test_metric
            print('Train {}'.format(train_metric))
            print('Dev   {}'.format(eval_metric))
            print('Test  {}\n'.format(test_metric))

    print('best dev {}: {}'.format(args.target_metric, best_result))
    print('best test {}: {}'.format(args.target_metric, best_test_metric))


    eval_metric = evaluate(eval_dataloader)
    log_path = args.log_path
    with open(log_path, 'a') as f:
        f.write('{} {} prefix lm\n'.format(args.task_name, args.dataset_name))
        f.write(f'seed: {args.seed}, batch_size: {args.per_device_train_batch_size}, lr: {args.learning_rate}, rank: {args.rank}, lora_alpha: {args.lora_alpha}, label_schema: {args.label_schema}, update_emb {args.update_emb}, task parameters: {trainable_parameters}\n')
        f.write(f'best test metric: {best_test_metric}\n')
        f.write(f'best epoch:{best_epoch}, {args.target_metric}, train: {train_metric[args.target_metric]}, dev: {best_result}, test {best_test_metric[args.target_metric]}\n\n')
        # f.write(f'task: {args.task_name}, seed: {args.seed}, batch_size: {args.per_device_train_batch_size}, lr: {args.learning_rate}, prefix_len: {args.prefix_len}, prefix_dropout: {args.prefix_dropout}, mid_dim: {args.mid_dim}, label_schema: {args.label_schema} add_special_token: {args.add_special_token}\nbest epoch:{best_epoch}, best_dev_result: {args.target_metric} {best_result}\n\n')
    # hyperlogger.save_results('./hyperTune/logs/{}_{}_log_{}.json'.format(args.task_name, args.dataset_name, '_'.join(time.asctime( time.localtime(time.time()) ).split())))


class DataCollatorForLMTokanClassification(DataCollatorForTokenClassification):
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        ori_labels = [feature['ori_labels'] for feature in features] if 'ori_labels' in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            batch['ori_labels'] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in ori_labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            batch["ori_labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in ori_labels]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch

def add_label_token_roberta(model, tokenizer, label_map):
    sorted_add_tokens = sorted(list(label_map.keys()), key=lambda x: len(x), reverse=True)
    tokenizer.add_tokens(sorted_add_tokens)
    num_tokens, _ = model.roberta.embeddings.word_embeddings.weight.shape
    model.resize_token_embeddings(len(sorted_add_tokens)+num_tokens)
    for token in sorted_add_tokens:
        if token.startswith('B-') or token.startswith('I-'):
            index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            if len(index)>1:
                raise RuntimeError(f"{token} wrong split")
            else:
                index = index[0]
            assert index>=num_tokens, (index, num_tokens, token)
            if isinstance(label_map[token], list):
                indexes = tokenizer.convert_tokens_to_ids(label_map[token])
            else:
                indexes = tokenizer.convert_tokens_to_ids([label_map[token]])
            embed = model.roberta.embeddings.word_embeddings.weight.data[indexes[0]]
            for i in indexes[1:]:
                embed += model.roberta.embeddings.word_embeddings.weight.data[i]
            embed /= len(indexes)
            model.roberta.embeddings.word_embeddings.weight.data[index] = embed
    return tokenizer

def add_label_token_bert(model, tokenizer, label_map):
    sorted_add_tokens = sorted(list(label_map.keys()), key=lambda x: len(x), reverse=True)
    tokenizer.add_tokens(sorted_add_tokens)
    num_tokens, _ = model.bert.embeddings.word_embeddings.weight.shape
    model.resize_token_embeddings(len(sorted_add_tokens)+num_tokens)
    for token in sorted_add_tokens:
        if token.startswith('B-') or token.startswith('I-'):  # 特殊字符
            index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            if len(index)>1:
                raise RuntimeError(f"{token} wrong split")
            else:
                index = index[0]
            assert index>=num_tokens, (index, num_tokens, token)
            indexes = tokenizer.convert_tokens_to_ids([label_map[token]])
            embed = model.bert.embeddings.word_embeddings.weight.data[indexes[0]]
            for i in indexes[1:]:
                embed += model.bert.embeddings.word_embeddings.weight.data[i]
            embed /= len(indexes)
            model.bert.embeddings.word_embeddings.weight.data[index] = embed
    return tokenizer

def mask_entity_batch(batch, tokenizer, mask_prob, label_to_id):
    # 只针对当前batch的entity按概率进行mask
    ori_input_ids = batch['input_ids']
    ori_label_ids = batch['ori_labels']

    # ori_input_ids, input_mask, segment_ids, label_ids, subword_mask, ori_label_ids = batch
    input_ids = ori_input_ids.clone().detach()
    entity_index = ori_label_ids != label_to_id['O']
    entity_mask = torch.full(entity_index.shape, 0.0)
    entity_mask[entity_index] = mask_prob
    entity_prob_mat = torch.bernoulli(entity_mask).bool()
    input_ids[entity_prob_mat] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    batch['input_ids'] = input_ids
    return batch

if __name__ == "__main__":
    main()