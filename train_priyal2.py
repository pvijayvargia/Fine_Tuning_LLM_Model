import os

#Importing necessary libraries
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from datasets import load_dataset
import torch

#Importing custom modules
import bitsandbytes as bnb
from huggingface_hub import login, HfFolder
from trl import SFTTrainer
from utils import print_trainable_parameters, find_all_linear_names
from train_args_priyal import ScriptArguments
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training

#Parsing the script arguments using HfArgumentParser and ScriptArguments data class
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

#Function to perform training
def training_function(args):

    #Authenticate with the HuggingFace API using the provided token
    login(token=args.hf_token)
    
    #Setting the seed
    set_seed(args.seed)

    #Load data set from the specified data path
    data_path=args.data_path

    dataset = load_dataset(data_path)

    #Configure bitsandbytes (bnb) quantization for low-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    #Load pre-trained model with bnb quantization config
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        use_cache=False,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    #Load the tokenizer associated with the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'

    #Prepare model for training
    model=prepare_model_for_kbit_training(model)

    #Find all linear layer names in the model
    modules=find_all_linear_names(model)

    #Configure LoRA with peftConfig
    config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=modules
    )

    #Add LoRA adapters to the model
    model=get_peft_model(model, config)

    #Define training arguments
    output_dir = args.output_dir
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        bf16=False,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        tf32=False,
        report_to="none",
        push_to_hub=False,
        max_steps = args.max_steps
    )
    #Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'].select(range(2000)), #Specifying the data to take only first 2000 rows
        dataset_text_field=args.text_field,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments
    )

    #Converter batch normalization layers to float 32 for compatibility with bnb
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    print('starting training')

    #Train the model
    trainer.train()

    #Training completed, save LoRA adapters to the specified directory
    print('LoRA training complete')
    lora_dir = args.lora_dir
    trainer.model.push_to_hub(lora_dir, safe_serialization=False)
    
    print("saved lora adapters")

    
#Entry point for script execution
if __name__=='__main__':
    training_function(args)

