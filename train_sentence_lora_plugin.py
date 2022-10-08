import argparse
import logging
import math
import os
import random
import torch
from pathlib import Path

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

import loralib as lora
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

import loralib as lora


logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def get_logger(filename, verbosity=1, name=__name__):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='cola',
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
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
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
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
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
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
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
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
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    
    parser.add_argument(
        "--log_path", type=str, default='/root/plugin_tuning/log/log.txt')
    
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--rank', type=int, default=1)
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

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

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
    logger = get_logger(filename='/root/plug_lora/log/bug_log/{}.log'.format(args.task_name))
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

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, add_prefix_space=True)
    # create label word map

    # from model.modeling_robert_lora import RobertaForMaskedLM
    # config.rank = args.rank
    # config.lora_alpha = args.lora_alpha

    # if 'roberta' in args.model_name_or_path:

    from model.modeling_robert_lora import RobertaForMaskedLM
    config.rank = args.rank
    config.lora_alpha = args.lora_alpha
    model = RobertaForMaskedLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config
    )
    # model = AutoModelWithHeads.from_pretrained(args.model_name_or_path, config=config)
    mask_token = '<mask>'
    # mask_idx = tokenizer.convert_tokens_to_ids(mask_token)
    mask_idx = tokenizer.convert_tokens_to_ids(mask_token)
    
    
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    # )

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    target_metric_map = {
        'cola':'matthews_correlation',
        'sst2':'accuracy',
        'mrpc': 'f1',
        'qqp': 'f1',
        'mnli': 'accuracy',
        'qnli': 'accuracy',
        'rte': 'accuracy'
    }

    label_word_dict = {
        'cola': {0:'wrong', 1:'correct'},
        'sst2': {0:'Ġgreat', 1:'Ġterrible'},
        'mrpc': {0:'Ġdifferent', 1:'Ġsimilar'},
        'qqp': {0: 'Ġmixed', 1:'Ġtwo'}, 
        'mnli': {2: 'Ġlol', 0: 'ĠâĢĶ', 1: 'âĢĭ'},
        # 'mnli': {0: 'Yes', 1: 'No', 2: 'Maybe'},
        'qnli': {0: 'ĠIn', 1: 'Ġ??'},
        'rte': {1: 'Ġ??', 0: 'ĠYes'},
        # 'boolq': {True:'Yes', False:'No'},
        # 'cb': {'entailment':'Yes', 'contradiction': 'No', 'nautral': 'Maybe'},
    }

    
    label_word_map = {key:tokenizer.convert_tokens_to_ids(value) for key, value in label_word_dict[args.task_name].items()}
    label_word_map[-1] = -1
    target_metric = target_metric_map[args.task_name]

    

    padding = "max_length" if args.pad_to_max_length else False
    padding = "max_length"

    def preprocess_function(examples):
        # Tokenize the texts
        # texts = (
        #     (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        # )
        texts = add_prompt(args.task_name, examples, sentence1_key, sentence2_key, mask_token)

        result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
        
        if 'label' in examples:
            labels = []
            label_position_idx = []
            label_word_idx_list = [label_word_map[item] for item in examples["label"]]
            for idx, token_ids_list in enumerate(result['input_ids']):
                label_word_ids = label_word_idx_list[idx]
                label_position_idx.append(token_ids_list.index(mask_idx))
                labels.append([-100 if item!=mask_idx else label_word_ids for item in token_ids_list])
            
            result['labels'] = labels
        else:
            label_position_idx = []
            label_word_idx_list = [label_word_map[item] for item in examples["label"]]
            for idx, token_ids_list in enumerate(result['input_ids']):
                label_position_idx.append(token_ids_list.index(mask_idx))
        result['label_position_idx'] = label_position_idx

        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False
        )

    train_dataset = processed_datasets["train"]
    # train_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    dev_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    # test_dataset = 
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    dev_dataloader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    lora.mark_only_lora_as_trainable(model, bias='lora_only')
    
    allow_optimize_list = ['lora']
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

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable parameters: {}'.format(trainable_parameters))
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    args.num_warmup_steps = args.warmup_ratio * args.max_train_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_epoch = 0
    best_metric = {target_metric:0}
    for epoch in range(args.num_train_epochs):
        epoch_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            label_position_idx = batch.pop('label_position_idx')
            outputs = model(**batch)
            loss = outputs.loss
            epoch_loss += loss.item()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_description('loss: %.4f'%(epoch_loss/(step+1)))
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        # accelerator.wait_for_everyone()
        model.eval()
        for step, batch in enumerate(dev_dataloader):
            label_position_idx = batch.pop('label_position_idx')
            outputs = model(**batch)
            predictions, labels = extract_real_label(label_position_idx, outputs.logits, batch['labels'], label_word_map)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(labels),
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")
        if eval_metric[target_metric] > best_metric[target_metric]:
            best_metric = eval_metric
            best_epoch = epoch

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )
    # if accelerator.is_local_main_process:
    with open(args.log_path, 'a') as f:
        f.write('{} adapter_lm\n'.format(args.task_name))
        f.write(f'seed: {args.seed}, batch_size: {args.per_device_train_batch_size}, lr: {args.learning_rate}, task parameters: {trainable_parameters}\n')
        f.write(f'best epoch:{best_epoch}, {target_metric}, dev: {best_metric}\n\n')
    

        


def add_prompt(task_name, example, sentence1_key, sentence2_key, mask_token='<mask>', max_length=128):
    sentence1_list = example[sentence1_key]
    if sentence2_key != None:
        sentence2_list = example[sentence2_key]
    if task_name=='sst2':
        # return [item for item in sentence1_list]
        # return [' '.join([item, mask_token])]
        return [' '.join(['It', 'was', mask_token, '.', item]) for item in sentence1_list]
    elif task_name=='boolq':
        return [' '.join([mask_token, item1, item2]) for item1, item2 in zip(sentence1_list, sentence2_list)]
    elif task_name == 'cb':
        return [' '.join([item1, mask_token, item2]) for item1, item2 in zip(sentence1_list, sentence2_list)]
    elif task_name == 'mnli':
        return [''.join([mask_token, '?', item1,item2]) for item1, item2 in zip(sentence1_list, sentence2_list)]
        # return [' '.join([item1, mask_token, item2]) for item1, item2 in zip(sentence1_list, sentence2_list)]
    elif task_name == 'mrpc':
        # return [' '.join([item1, 'and', item2, 'have', mask_token, 'meanings.']) for item1, item2 in zip(sentence1_list, sentence2_list)]
        # return [' '.join([item1, 'and', item2, 'have', mask_token, 'meanings.']) for item1, item2 in zip(sentence1_list, sentence2_list)]
        # return [' '.join([item1, 'and', item2, 'have', mask_token, 'meanings.']) for item1, item2 in zip(sentence1_list, sentence2_list)]
        return [' '.join([mask_token, item1, item2]) for item1, item2 in zip(sentence1_list, sentence2_list)]
    elif task_name == 'qnli':
        # return [' '.join([item1, '?', mask_token, ',', item2]) for item1, item2 in zip(sentence1_list, sentence2_list)]
        return [' '.join([mask_token, '?', item1, item2]) for item1, item2 in zip(sentence1_list, sentence2_list)]
    elif task_name == 'qqp':
        # return [' '.join([item1, item2, mask_token]) for item1, item2 in zip(sentence1_list, sentence2_list)]
        return [' '.join([mask_token, item1, item2]) for item1, item2 in zip(sentence1_list, sentence2_list)]
    elif task_name == 'rte':
        # return [' '.join([item1, mask_token, item2]) for item1, item2 in zip(sentence1_list, sentence2_list)]
        return [' '.join([mask_token, item1, item2]) for item1, item2 in zip(sentence1_list, sentence2_list)]
    elif task_name == 'cola':
        # return [' '.join([item, 'Grammar is', mask_token]) for item in sentence1_list]
        return [' '.join(['Grammar is', mask_token, item]) for item in sentence1_list]
    elif task_name == 'stsb':
        return [' '.join([item1, item2, mask_token]) for item1, item2 in zip(sentence1_list, sentence2_list)]
    else:
        pass

def extract_real_label(label_position_idx, output_logits, labels, label_word_map):
    # output_logits (batch_size, seq_len, vocab_size)
    label_word_predictions = torch.diag(torch.index_select(output_logits.argmax(dim=-1), 1, label_position_idx))
    labels = torch.diag(torch.index_select(labels, 1, label_position_idx))
    for key, value in label_word_map.items():
        if value == -1:
            continue
        key_mat = torch.full(labels.shape, key).type_as(labels)
        labels = torch.where(labels==value, key_mat, labels)
        label_word_predictions = torch.where(label_word_predictions==value, key_mat, label_word_predictions)

    return label_word_predictions, labels

def extract_predict_label(label_position_idx, output_logits, label_word_map):
    label_word_predictions = torch.diag(torch.index_select(output_logits.argmax(dim=-1), 1, label_position_idx))
    reverse_label_word_map = {v:k for k,v in label_word_map.items()}
    for i in range(label_word_predictions.shape[0]):
        if label_word_predictions[i] in reverse_label_word_map:
            label_word_predictions[i] = reverse_label_word_map[label_word_predictions[i]]
        else:
            label_word_predictions[i] = 0
    return label_word_predictions




if __name__ == "__main__":
    main()