#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import json
import io


from uuid import uuid4
import logging
import math
import os
import subprocess
from datetime import datetime
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import datasets
import torch
from build_dataset import build_instruction_dataset, DataCollatorForSupervisedDataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX = -100

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "sft_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        # kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    # def on_train_end(self, args, state, control, **kwargs):
    #     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     peft_model_path = os.path.join(args.output_dir , "sft_lora_model")
            
    #     kwargs["model"].save_pretrained(peft_model_path)
        
    #     kwargs["tokenizer"].save_pretrained(peft_model_path)
    
    def on_train_end(self, args, state, control, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir =args.output_dir  + "_" + timestamp
        peft_model_path = os.path.join(output_dir, "sft_lora_model")
        merged_path = args.output_dir  + "_" + timestamp + "_merged_model"
        # 保存lora模型
        kwargs["model"].save_pretrained(peft_model_path)
        
        kwargs["tokenizer"].save_pretrained(peft_model_path)
        
        # 合并模型
        command = f"python /home/public/zp-codes/scripts/training/chinese-alpaca-2-7b/encrypt/merge_llama2_with_chinese_lora_low_mem.py" \
                    f" --base_model {args.pretrained_model}" \
                    f" --lora_model {peft_model_path}" \
                    " --output_type huggingface" \
                    f" --output_dir {merged_path}"

        try:
            subprocess.run(command, shell=True)
        except subprocess.CalledProcessError as e:
            error_msg = f'执行脚本出错，错误码：{e.returncode}，错误信息：{e.output.decode("utf-8")}'
            print(error_msg)
        
        # 生成UUID
        uuid = str(uuid4())
        training_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_version = 'version ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        # 构建新记录
        new_record = {
            "uuid": uuid,
            "training_time": training_time,
            "model_path": merged_path,
            "model_version": model_version,
            "loss_file": f"{peft_model_path}/trainer_state.json",
        }

        # 读取现有JSON文件内容
        with open("/home/public/zp-codes/logs/training.json", "r") as file:
            data = json.load(file)

        # 追加新记录
        data.append(new_record)

        # 写回到文件
        with open("/home/public/zp-codes/logs/training.json", "w") as file:
            json.dump(data, file, indent=4)
        

# class EvaluateAccuracyCallback(transformers.TrainerCallback):
    
#     def __init__(self, eval_dataset,):
#         self.eval_dataset = eval_dataset


#     def on_step_end(self, args, state, control, **kwargs):
#         if state.global_step % 100 == 0:
#             # 获取当前模型
#             model = kwargs["model"]
#             tokenizer = kwargs['tokenizer']
#             print(model)
#             # 在评估模式下计算准确率
#             model.eval()
#             print('callback self.eval_dataset',self.eval_dataset)
#             correct_predictions = 0
            
#             eval_dataset = self.eval_dataset.remove_columns(['id'])
#             print(eval_dataset)
#             for eval_example in eval_dataset:
#                 with torch.no_grad():
#                     outputs = model(input_ids = eval_example['input_ids'].unsqueeze(0).to("cuda"))
    
#                 predictions = torch.argmax(outputs.logits, dim=-1)
#                 print('predictions', predictions)
#                 print(outputs.logits.shape)
#                 label = eval_example['labels'].unsqueeze(0).to("cuda")
#                 print('label', label)
#                 correct_predictions += (predictions[0][-3] == label[0][-2] ).sum().item()
#             total_samples = len(eval_dataset)
#             accuracy = correct_predictions / total_samples
#             print(f"Step {state.global_step}: Evaluation Accuracy: {accuracy}")



# class EvaluateAccuracyCallback(transformers.TrainerCallback):
    
#     def __init__(self, eval_dataset,):
#         self.eval_dataset = eval_dataset


#     def on_step_end(self, args, state, control, **kwargs):
#         if state.global_step % 2 == 0:
#             # 获取当前模型
#             model = kwargs["model"]
#             tokenizer = kwargs['tokenizer']
#             print(model)
#             # 在评估模式下计算准确率
#             model.eval()
#             print('callback self.eval_dataset',self.eval_dataset)
#             correct_predictions = 0
            
#             eval_dataset = self.eval_dataset.remove_columns(['id'])
#             data_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True)
#             print(eval_dataset)
#             for batch in data_loader:
#                 with torch.no_grad():
#                     print('batch', batch)
#                     inputs = batch['inputs_ids'].to("cuda")
#                     labels = batch['labels'].to("cuda")
#                     outputs = model(**inputs)

#                 print(outputs.logits.shape)
#                 predictions = torch.argmax(outputs.logits, dim=-1)
#                 print('predictions', predictions)
#                 print('labels', labels)
#                 correct_predictions += (predictions[:][-3] == label[:][-2] ).sum().item()
#             total_samples = len(eval_dataset)
#             accuracy = correct_predictions / total_samples
#             print(f"Step {state.global_step}: Evaluation Accuracy: {accuracy}")




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default=None, metadata={"help": "The datasets processed stored"})

    max_seq_length: Optional[int] = field(default=1024)


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    pretrained_model: Optional[str] = field(default=None)
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    peft_path : Optional[str] = field(default=None)
    flash_attn : Optional[bool] = field(default=False)


logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.flash_attn:
        from flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)


    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if (len(tokenizer))!=55296:
        raise ValueError(f"The vocab size of the tokenizer should be 55296, but found {len(tokenizer)}.\n"
                         "Please use Chinese-LLaMA-2 tokenizer.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset=None
    train_dataset = None

    if training_args.do_train:
        with training_args.main_process_first(desc="loading and tokenization"):
            path = Path(data_args.dataset_dir)
            files = [os.path.join(path,file.name) for file in path.glob("*.json")]
            logger.info(f"Training files: {' '.join(files)}")
            train_dataset = build_instruction_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                data_cache_dir = None,
                preprocessing_num_workers = data_args.preprocessing_num_workers)
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("Training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    if training_args.do_eval:
        with training_args.main_process_first(desc="loading and tokenization"):
            files = [data_args.validation_file]
            logger.info(f"Evaluation files: {' '.join(files)}")
            eval_dataset = build_instruction_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                data_cache_dir = None,
                preprocessing_num_workers = data_args.preprocessing_num_workers)
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("Evaluation example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            # 
            low_cpu_mem_usage=True,
            device_map = "auto"
        )
        # for name, param in model.named_parameters():
        #     if name == 'model.layers.31.mlp.up_proj.weight':
        #         print('random data', param.data)
        #         key = 'RtqikMWYdz45HAuXx-dguw6lmNXgOyT0XIMV8r0TtvU='
        #         with open(model_args.model_name_or_path + '/encrpt.pth', 'rb') as fr:
        #             encrypted_data = fr.read()
                    
        #         decrypted_data = Fernet(key).decrypt(encrypted_data)
        #         b = io.BytesIO(decrypted_data)
        #         b.seek(0)
        #         decrypted_state_dict = torch.load(b, map_location=param.device)
            
        #         param.data = decrypted_state_dict
        #         print('origian param:', param.data)
        #         # exit(0)
            
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"len(tokenizer):{len(tokenizer)}")
    if model_vocab_size != len(tokenizer):
        logger.info(f"Resize model vocab size to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    if training_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, training_args.peft_path)
    else:
        logger.info("Init new peft model")
        target_modules = training_args.trainable.split(',')
        modules_to_save = training_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
            modules_to_save=None
        lora_rank = training_args.lora_rank
        lora_dropout = training_args.lora_dropout
        lora_alpha = training_args.lora_alpha
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"lora_rank: {lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save)
        model = get_peft_model(model, peft_config)

    #model.base_model.tie_weights()
    model.print_trainable_parameters()
    logger.info(f"model.modules_to_save: {model.modules_to_save}")
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # trainer.add_callback(SavePeftModelCallback
    # trainer.add_callback(EvaluateAccuracyCallback(eval_dataset))
    
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        metrics["eval_samples"] =len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
