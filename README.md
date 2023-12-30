# Supercharge Message Summarization Experience: Parameter-Efficient Fine-Tuning and LLM Assisted Evaluation

## Introduction
In today's business landscape, we are surrounded by a wealth of opportunities to utilize advanced technology powered by AI. Think of large language models(LLMs) as versatile tools in our toolkit: we can *customize* them for a variety of specific downstream tasks, a process known as *fine-tuning*. However, a challenge arises in that each fine-tuned model typically maintains the same parameter size as the original. Therefore, managing multiple fine-tuned models requires careful consideration of factors such as accuracy performance, memory management, inference latency, and disk utilization.

Parameter-Efficient Fine-Tuning (PEFT) methods provide an efficient and streamlined approach for adapting pre-trained LLMs, commonly referred to as *base models*, to a range of specific downstream tasks. These tasks encompass diverse applications, including but not limited to text summarization, question answering, image generation, and text-to-speech synthesis. In contrast to traditional full fine-tuning, which consumes substantial computational resources, PEFT prioritizes the optimization of a significantly smaller parameter subset referred to as "adapters." These adapters work in tandem with the base model, achieving competitive performance while imposing lower computational and storage demands.

I've shared a [Colab notebook](https://github.com/bearbearyu1223/llm-fine-tuning-playground/blob/main/finetune_falcon_7b_conversation_summarization.ipynb) demonstrating a resource-efficient PEFT process using [QLoRA](https://arxiv.org/abs/2305.14314) and [HuggingFace PEFT libraries](https://github.com/huggingface/peft) to fine tune [Falcon-7B-sharded model](https://huggingface.co/vilsonrodrigues/falcon-7b-sharded) on [SamSum dataset](https://huggingface.co/datasets/samsum) for summarizing "message-like" conversations. It achieves reasonable summarization performance after training for only 5 epochs on an A100 compute instance with a single GPU. Additionally, I've employed `GPT-3.5-turbo` to assess generated summaries, showcasing a potentially automated evaluation method by formalizing evaluation guidelines into a prompt template. This approach stands in contrast to traditional automated evaluation metrics like ROUGE or BERTScore, which rely on reference summaries.

Furthermore, I will also share some insights and lessons I've gained throughout this process, with a particular focus on considerations when leveraging LLMs to develop product experiences related to summarization.

I hope you'll discover this article both informative and intriguing, igniting your creativity as you explore the development of your unique product experiences and strategies through the use of fine-tuned foundation models. 

Enjoy the read, and let your innovation flourish. Happy new year!

## Fine-Tuning with Model Quantization and LoRA
Base models such as Claude, T5, Falcon, and Llama2 excel at predicting tokens in sequences, but they *struggle with generating responses that align with instructions*. Fine-tuning techniques, such as **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning from Human Feedback (RLHF)**, can be employed to bridge these gaps. In this sample project, we'll explore the application of SFT to Falcon-7B, a 7-billion-parameter causal decoder model trained by TII on 1,500-billion tokens from RefinedWeb with curated corpora, for conversation summarization tasks.

### Install and Import the Required Libraries 
To get started, one can create a virtual environment and install all the required libraries needed for this sample project. In Colab, this can be done by running a cell containing the following scripts:
```Shell 
!pip install huggingface_hub==0.19.4
!pip install -q -U trl accelerate git+https://github.com/huggingface/peft.git
!pip install transformers==4.36.0
!pip install datasets==2.15.0 Tokenizers==0.15.0
!pip install -q bitsandbytes wandb
!pip install py7zr
```
then the installed libraries can be imported and be used during runtime via:
```Python
import torch
import numpy as np
from huggingface_hub import notebook_login
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training, TaskType
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
```
### Prepare the Dataset for Fine-Tuning 
You can load the [SamSum dataset](https://huggingface.co/datasets/samsum) directly using the [Hugging Face Datasets libraries](https://huggingface.co/docs/datasets/index via Python code:

```Python
dataset_name = "samsum"
dataset = load_dataset(dataset_name)

train_dataset = dataset['train']
eval_dataset = dataset['validation']
test_dataset = dataset['test']
dataset
```
The dataset contains a total of 14,732 samples for training, 818 samples for validation, and 818 samples for testing. A sample of the dataset is displayed below:
![sample_data](assets/dataset.png)

To format the original training dataset into prompts for instruction fine-tuning, you can use the following helper function. For more details, refer to the detailed reference [here](https://huggingface.co/docs/trl/sft_trainer#format-your-input-prompts)).

```Python
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['id'])):
        instruction = "Summarize this Dialogue."
        input = example['dialogue'][i]
        output = example['summary'][i]
        prompt="### Instruction:\n{instruction}\n\n### Dialogue:\n{input}\n\n### Summary:\n{output}".format(instruction=instruction, input=input, output=output)
        output_texts.append(prompt)
    return output_texts
```
### Set up the Configuration for Fine-Tuning 
To reduce VRAM usage during training, you will fine-tune [a resharded version of Falcon-7B](https://huggingface.co/vilsonrodrigues/falcon-7b-sharded) in 4-bit precision using [QLoRA](https://arxiv.org/abs/2305.14314). You can use the following code snippet to load the base model and prepare it for the QLoRA experiment:

```Python
model_name = "vilsonrodrigues/falcon-7b-sharded"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
```
Based on the QLoRA paper, we will taget all linear transformer block layers as target modules for fine-tuning (also see the discussions on reddit [here](https://www.reddit.com/r/LocalLLaMA/comments/15sgg4m/what_modules_should_i_target_when_training_using/?rdt=53925)). You can leverage the following helper function to find these target modules:

```Python
def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)
target_modules = find_target_modules(model)
print(target_modules)
```
And in this case, the target modules for fine-tuning will be 
`['dense_4h_to_h', 'dense_h_to_4h', 'query_key_value', 'dense']`. 

After loading and preparing the base model for QLoRA, you can configure the fine-tuning experiment using the following code:
```Python
model = prepare_model_for_kbit_training(model)

lora_alpha = 32 
lora_dropout = 0.1 
lora_rank = 16

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_rank,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
```
This configuration will result in an *adapter model* with *32,636,928 trainable parameters*, which is only *0.47%* of the trainable parameters compared to the *6,954,357,632 parameters* of the base model.
### Set up the Configuration for Trainig
Load the tokenizer from the pre-trained base model, both the base model, the LoRA config, and the tokenizer will be needed for the SFT trainer. 
```Python
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if tokenizer.pad_token_id is None:
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
```
Below is the configuration used for SFT 
```Python
output_dir = "falcon_7b_LoRA_r16_alpha32_epoch10_dialogue_summarization_v0.1"
per_device_train_batch_size = 32 #4
gradient_accumulation_steps = 4
gradient_checkpointing=False
optim = "paged_adamw_32bit"
save_steps = 20
logging_steps = 20
learning_rate = 2e-4
max_grad_norm = 0.1
warmup_ratio = 0.01
lr_scheduler_type = "cosine" #"constant"
num_train_epochs = 5
seed=42
max_seq_length = 512

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    num_train_epochs=num_train_epochs,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    push_to_hub=True,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    formatting_func=formatting_prompts_func,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    args=training_arguments,
)
```
You can initiate the fine-tuning experiment via
```Python
trainer.train()
```
The entire training process took approximately 3 hours running on an A100 instance with a single GPU.

### Model Inference of the Fined-Tuned Model 
Upon completion of the training process, you can easily share the adapter model by uploading it to Hugging Face's model repository using the following code:
```Python
trainer.push_to_hub() 
```
This published adapter model can then be retrieved and used in conjunction with the base model for various summarization tasks, as demonstrated in the reference code snippet below.
```Python 
PEFT_MODEL = "bearbearyu1223/falcon_7b_LoRA_r16_alpha32_epoch10_dialogue_summarization_v0.1"
config = PeftConfig.from_pretrained(PEFT_MODEL)
peft_base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

peft_model = PeftModel.from_pretrained(peft_base_model, PEFT_MODEL)

# Generate Summarization
def get_summary(dialogue, max_new_tokens=50, max_length=512, verbose=False):
  prompt= "### Instruction:\n{instruction}\n\n### Dialogue:\n{dialogue}\n\n### Summary:\n".format(instruction="Summarize the Dialogue below.", dialogue=dialogue)
  if verbose:
    print(prompt)

  peft_encoding = peft_tokenizer(prompt, truncation=True, return_tensors="pt").to(torch.device("cuda:0"))
  peft_outputs = peft_model.generate(input_ids=peft_encoding.input_ids, generation_config=GenerationConfig(max_length=max_length, do_sample=True,
                                                                                                         max_new_tokens=max_new_tokens,
                                                                                                         pad_token_id = peft_tokenizer.eos_token_id,
                                                                                                         eos_token_id = peft_tokenizer.eos_token_id,
                                                                                                         attention_mask = peft_encoding.attention_mask,
                                                                                                         temperature=0.1, top_k=1, repetition_penalty=30.0, num_return_sequences=1,))
  peft_text_output = peft_tokenizer.decode(peft_outputs[0], skip_special_tokens=True)

  sub = "### Summary:"
  raw_summary = peft_text_output.split(sub)[1]

  return raw_summary
```
See an example of a summary generated by the fine-tuned model in comparison to the reference summary crafted by a human below
```Python
test_index=6
dialogue=test_dataset[test_index]['dialogue']
summary=test_dataset[test_index]['summary']
peft_output=get_summary(dialogue,verbose=True)

print("Human Summary:")
print(summary)
print("PEFT Summary:")
print(peft_output)
```

| Instruction                   |
| ----------------------------- |
| Summarize the Dialogue below. |


| Speaker | Utterance                                           |
| ------- | --------------------------------------------------- |
| Max     | Know any good sites to buy clothes from?            |
| Payton  | Sure :) <file_other> <file_other> ... <file_other> |
| Max     | That's a lot of them!                               |
| Payton  | Yeah, but they have different things so I usually buy things from 2 or 3 of them. |
| Max     | I'll check them out. Thanks.                        |
| Payton  | No problem :)                                       |
| Max     | How about u?                                        |
| Payton  | What about me?                                      |
| Max     | Do u like shopping?                                 |
| Payton  | Yes and no.                                         |
| Max     | How come?                                           |
| Payton  | I like browsing, trying on, looking in the mirror and seeing how I look, but not always buying. |
| Max     | Y not?                                              |
| Payton  | Isn't it obvious? ;)                                |
| Max     | Sry ;)                                              |
| Payton  | If I bought everything I liked, I'd have nothing left to live on ;) |
| Max     | Same here, but probably different category ;)      |
| Payton  | Lol                                                 |
| Max     | So what do u usually buy?                          |
| Payton  | Well, I have 2 things I must struggle to resist!  |
| Max     | Which are?                                          |
| Payton  | Clothes, ofc ;)                                    |
| Max     | Right. And the second one?                         |
| Payton  | Books. I absolutely love reading!                  |
| Max     | Gr8! What books do u read?                         |
| Payton  | Everything I can get my hands on :)                |
| Max     | Srsly?                                              |
| Payton  | Yup :)                                              |

| Summary Type | Summary Description                                             |
| ------------ | ----------------------------------------------------------------- |
| Human        | Payton provides Max with websites selling clothes. Payton likes browsing and trying on the clothes but not necessarily buying them. Payton usually buys clothes and books as he loves reading. |
| PEFT         | Payton sends Max some links with online shops where she buys her stuff. Payton likes both fashion items and literature. She reads all kinds of fiction. |





## Evaluation of Summarization Quality 
Traditional approaches to evaluating summarization tasks typically involve metrics such as ROUGE (Recall-Oriented Understudy for Gisting Evaluation) and BLEU (Bilingual Evaluation Understudy) which assess the quality of generated summaries by comparing them to human-written reference summaries. These metrics measure factors like overlap in n-grams and word sequences, providing a quantitative assessment of summary quality.

When evaluating the quality of summarization generated from the fine-tuned LLM, it is crucial to develop clear and consistent annotation guidelines to provide instructions to the human annotators. Below is a list of metrics that we will consider:

### Metric 1: Relevance
**Capturing the Essence:** The LLM will assist annotators in evaluating the relevance of a summary. Annotators will evaluate the relevance of a summary on a scale of 1 to 5, considering whether the summary effectively extracts important content from the source conversation, avoiding redundancies and excess information. With clear criteria and steps, annotators can confidently assign scores that reflect the summary's ability to convey essential details.

### Metric 2: Coherence
**Creating Clarity:** The LLM will assist annotators in evaluating the coherence of a summary. Annotators will rate summaries from 1 to 5, focusing on the summary's organization and logical flow. Clear guidelines enable annotators to determine how well the summary presents information in a structured and coherent manner.

### Metric 3: Consistency
**Factually Sound:** The LLM will assist annotators in evaluating the consistency of a summary. Annotators will assess summaries for factual alignment with the source conversation, rating them from 1 to 5. SummarizeMaster ensures that annotators identify and penalize summaries containing factual inaccuracies or hallucinated facts, enhancing the reliability of the evaluation process.

### Metric 4: Fluency
**Language Excellence:** The LLM will assist annotators in evaluating the fluency of a summary. Fluency is a critical aspect of summary evaluation. Annotators will assess summaries for grammar, spelling, punctuation, word choice, and sentence structure, assigning scores from 1 to 5.

These instructions are provided as a prompt template to ensure consistent and data-driven evaluation of both human-generated and fine-tuned LLM summaries, aiming to enhance consistency, standardization, and efficiency in an otherwise labor-intensive manual evaluation process.


## Lessons Learned 
Fine-tuning capitalizes on pretraining knowledge but may falter when faced with data that the base model has never encountered in the training dataset. However, it can yield impressive results if that is not the case. 