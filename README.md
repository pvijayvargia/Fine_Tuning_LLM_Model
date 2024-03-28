# Fine-Tuning Llama 2 7B for Question Generation

## Introduction

This project focuses on fine-tuning the Llama 2 7B language model, developed by Meta, for the task of question generation. By fine-tuning Llama 2 7B on a dataset suitable for question generation, I aimed to create a model capable of generating high-quality and contextually relevant questions given input text or context.

## Dataset Information

For this project, I will utilized publicly available question-answer datasets suitable for question generation tasks. These datasets may include FAQs, trivia, or any other question-answer pairs sourced from reliable sources. The choice of dataset will depend on the specific application and requirements of the question generation task.

## Steps taken in the fine-tuning model

- **Fine-Tuning Script:** 
  - Utilized the Hugging Face Transformers library to fine-tune Llama 2 7B on the question generation dataset.
  - Adapted the provided fine-tuning script to incorporate dataset loading, model configuration, training loop, and evaluation metrics.

- **Fine-Tuning Hyperparameters:**
  - `learning_rate`: Control the learning rate during optimization to adjust the model's weights.
  - `batch_size`: Determine the number of samples processed in each training iteration.
  - `num_epochs`: Specify the number of training epochs for model convergence.
  - `max_seq_length`: Limit the maximum sequence length of input text processed by the model.
  - `task_name`: Define the task name as question generation.
  - `output_dir`: Specify the output directory to save the fine-tuned model and training logs.
  - `data_dir`: Provide the directory containing the question-answer dataset for fine-tuning.
  - `model_name_or_path`: Select the pre-trained LLM model (Llama 2 7B) to use as the base for fine-tuning.
  - `tokenizer_name`: Choose the tokenizer to preprocess text data for the model.

- **Fine-Tuning Hyperparameters:**
  - Fine-tuned Llama 2 7B on the question generation dataset using the provided script.
  - Saved the fine-tuned model to the designated output directory in hugging face

---
This personal project demonstrates my expertise in fine-tuning large language models for specific natural language processing tasks, focusing on question generation using Llama 2 7B.
