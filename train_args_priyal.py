from dataclasses import dataclass, field
import os
from typing import Optional

@dataclass
class ScriptArguments:

    hf_token: str = field(metadata={"help": "Hugging face access token to access hf platform"})


    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "specific model architecture in hf of the pre-trained model to be used."}
    )

    seed: Optional[int] = field(
        default=4761, metadata = {'help':'Sets seed for random number generator'}
    )

    data_path: Optional[str] = field(
        default="jbrophy123/quora_dataset", metadata={"help": "specifies the input data path."} 
    )

    output_dir: Optional[str] = field(
        default="output", metadata={"help": "specifies the directory where the ouput will be saved."}
    )
    
    per_device_train_batch_size: Optional[int] = field(
        default = 2, metadata = {"help":"sets batch size per GPU used during training. It controls the no. of samples processed in parallel on each GPU"}
    )

    gradient_accumulation_steps: Optional[int] = field(
        default = 1, metadata = {"help":"determines number of gradient accumulation steps before performing parameter updates during training."}
    )

    optim: Optional[str] = field(
        default = "paged_adamw_32bit", metadata = {"help":"specifies the optimiser for updating model parameters during training"}
    )

    save_steps: Optional[int] = field(
        default = 500, metadata = {"help":"specifies frequency for model checkpoints to be saved during training"}
    )

    logging_steps: Optional[int] = field(
        default = 1, metadata = {"help":"specifies frequency for logs to be recorded and displayed during training "}
    )

    learning_rate: Optional[float] = field(
        default = 2e-4, metadata = {"help":"initial learning rate used by optimizer. It controls the size of the parameter updates and the convergence speed of the training process"}
    )

    max_grad_norm: Optional[float] = field (
        default = 0.3, metadata = {"help":"maximum norm value for gradient clipping. Gradient clipping helps prevent exploding gradients and stabilizes training"}
    )

    num_train_epochs: Optional[int] = field (
        default = 1, metadata = {"help":"no. of times the entire dataset is processed during training"}
    ) 

    warmup_ratio: Optional[float] = field (
        default = 0.03, metadata = {"help":"ratio of warmup steps. Warmup is a technique used to gradually increase the learning rate at the begining of training to prevent instabilities."}
    )

    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata = {"help":"type of learning rate scheduler. Learning sschedulers adjust the learning rate during training according to predefined schedules."}
    ) 

    lora_dir: Optional[str] = field(default = "pvijayvargia/Fine-Tuning-LLM-Model", metadata = {"help":"directory for LoRA"})

    max_steps: Optional[int] = field(default=-1, metadata={"help": "maximum number of training steps. Training will stop once this limit is reached."})

    text_field: Optional[str] = field(default='chat_sample', metadata={"help": "specifies the name of the text field in the dataset. It is used to identify which field contains input text data for model training."})
    


