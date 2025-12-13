# Bias Mitigation and Evaluation for Large Language Models

This repository contains a complete pipeline to:

- Generate controlled synthetic text data
- Fine tune language models with LoRA and standard training
- Evaluate bias using custom metrics and statistical tests

It accompanies the research work:

> **Promoting Fairness in LLMs: Detection and Mitigation of Gender Bias**  
> Preprint on Research Square: https://www.researchsquare.com/article/rs-6461545/v1  
> DOI: https://doi.org/10.21203/rs.3.rs-6461545/v1  

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Installation](#installation)  
4. [Data Generation](#data-generation)  
5. [Training Workflows](#training-workflows)  
    - [LoRA training on GPT-2](#1-lora-training-on-gpt-2-training_modelpy)  
    - [Fine tuning GPT-Neo](#2-fine-tuning-gpt-neo-gpt3py)  
    - [LoRA training on Llama 3](#3-lora-training-on-llama-321b-train_llama3bpy)  
6. [Running a Fine Tuned Model](#running-a-fine-tuned-model)  
7. [Bias Evaluation and Metrics](#bias-evaluation-and-metrics)  
8. [Configuration and Environment Variables](#configuration-and-environment-variables)  
9. [Reproducing Results at a High Level](#reproducing-results-at-a-high-level)  
10. [Contributors](#contributors)  

## Project Overview

Large language models often learn and amplify social biases such as gender and racial bias.  
This project focuses on:

- Detecting bias in model outputs across sensitive factors such as gender, race, age and others  
- Measuring bias with quantitative metrics  
- Reducing bias through prompt engineering and LoRA based fine tuning  

The code targets both English and Hindi prompts and supports experiments similar to those in the paper.

## Repository Structure

| File or Folder | Description |
| ------------- | ----------- |
| [`custom_dataset_templates.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/custom_dataset_templates.py) | Generates a synthetic, fairness oriented text dataset from hand designed templates and completions. Outputs a CSV file. |
| [`unbiased_dataset.json`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/unbiased_dataset.json) | Example unbiased dataset for experiments. Content aligns with the templates in the generator. |
| [`gpt3.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/gpt3.py) | Fine tunes a GPT-Neo model on JSONL dialog data using Hugging Face Trainer. Saves a standard causal language modeling checkpoint. |
| [`training_model.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/training_model.py) | LoRA training pipeline for GPT-2 using PEFT. Loads JSONL data with `input` and `output` fields and saves LoRA weights and an optional merged model. |
| [`train_llama3b.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/train_llama3b.py) | LoRA training pipeline for `meta-llama/Llama-3.2-1B` using PEFT. Very similar to `training_model.py` but for Llama. |
| [`running_finetuned_model.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/running_finetuned_model.py) | Loads a fine tuned GPT-2 model from disk and generates an answer to a bias related question. Useful for quick sanity checks. |
| [`ics_testing.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/ics_testing.py) | Computes Idea Consistency Score (ICS) using a SentenceTransformer model and a fine tuned language model, and plots percentage scores for different factors. |
| [`zero_shot_testing.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/zero_shot_testing.py) | Sets up a zero shot classification pipeline and OpenAI client to analyze bias across labels such as gender, race and portrayal. Intended to produce ZSC based bias scores and visualizations. |
| [`t_test.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/t_test.py) | Uses simulated samples around reported means and runs t tests for the ZSC and thematic consistency tables from the paper. Prints tables with p values. |
| [`Metrics_Eval.ipynb`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/Metrics_Eval.ipynb) | Jupyter notebook for computing metrics such as DI, ICS, TCS, implicit bias style scores and ZSC, along with plots. |
| [`readme.md`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/readme.md) | Original short README for the project. |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tejanshsachdeva/Biasness-Mitigation.git
cd Biasness-Mitigation
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux or macOS
# .venv\Scripts\activate  # Windows PowerShell
```

### 3. Install Python dependencies

There is no `requirements.txt` in the repository. A typical environment for all scripts includes:

```bash
pip install     torch     transformers     datasets     peft     sentence-transformers     matplotlib     pandas     numpy     python-dotenv     scipy     openai
```

You also need:

- Access to the Hugging Face models used in the scripts  
- An OpenAI API key set in your environment for `zero_shot_testing.py`  

## Data Generation

### `custom_dataset_templates.py`

This script builds a synthetic dataset that discourages biased reasoning.

Key ideas:

- Uses lists of
  - Occupations such as doctor, nurse, engineer, teacher, CEO, athlete, artist  
  - Traits such as intelligent, nurturing, strong, innovative, hardworking  
  - Countries such as India, USA, Mexico, Nigeria, China, Germany  
  - Names such as Aisha, Carlos, Lakisha, John, Wei, Fatima  
  - Pronoun groups such as masculine, feminine, neutral  

- Combines these with templates such as  
  - `{pronoun} is a skilled {occupation}.`  
  - `{name} from {country} is known for {trait}.`  

- Uses completions that explicitly state neutral or fairness aligned statements.  

Each example is stored as a single text field that looks like:

```text
System: You are an AI that provides unbiased responses.
User: <prompt>
Assistant: <response>
```

The main entry point at the end of the file:

```python
dataset = generate_diverse_dataset(size=1000)
save_to_csv(dataset)
```

This will create `diverse_dataset.csv` in the current directory.

You can change the size and file name by editing the function arguments.

## Training Workflows

You can train with three main scripts:

- LoRA based GPT-2 training  
- GPT-Neo fine tuning  
- LoRA based Llama 3 point 2 billion training  

All training scripts assume you have a GPU, but they also work on CPU with lower speed.

### 1. LoRA training on GPT-2 (`training_model.py`)

File: [`training_model.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/training_model.py)

This script:

1. Reads JSONL files from a data directory  
2. Wraps GPT-2 with a LoRA configuration using PEFT  
3. Trains with Hugging Face Trainer on a causal language modeling objective  
4. Saves LoRA weights and an optional merged model  

#### Data format

Each line in the JSONL files should look like:

```json
{"input": "user prompt text ...", "output": "assistant reply text ..."}
```

The script formats each line internally as:

```text
### User: <input>

### Assistant: <output>
```

#### Paths and environment variables

At the top of the file:

- `BASE_DIR` is read from the `BASE_DIR` environment variable if present, otherwise uses a default Windows path  
- `DATA_DIR = os.path.join(BASE_DIR, "data")`  
- `OUTPUT_DIR = os.path.join(BASE_DIR, "modeel")`  
- Inside `OUTPUT_DIR` it creates
  - `lora_weights` for LoRA only
  - `merged_model` for merged weights
  - `logs` for training logs

You can set:

```bash
export BASE_DIR=/path/to/your/project
export MODEL_NAME=gpt2
export LORA_R=8
export LORA_ALPHA=16
export LORA_DROPOUT=0.05
export BATCH_SIZE=4
export GRAD_ACCUMULATION=4
export LEARNING_RATE=2e-4
export NUM_EPOCHS=3
export MAX_LENGTH=512
```

On Windows, use `set` instead of `export`.

#### Running the trainer

1. Place your JSONL files in `${BASE_DIR}/data`  
2. Run:

```bash
python training_model.py
```

At the end you will see:

- LoRA checkpoint in `OUTPUT_DIR/lora_weights`  
- Optional merged model in `OUTPUT_DIR/merged_model`  
- Config file `training_config.json` that records hyperparameters  

You can adapt this setup to run bias specific training experiments.

### 2. Fine tuning GPT-Neo (`gpt3.py`)

File: [`gpt3.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/gpt3.py)

This script fine tunes `EleutherAI/gpt-neo-125m` on dialog style data.

#### Data format

The script expects a JSONL file where each line is a list of dialog turns, for example:

```json
[
  {"role": "user", "content": "Text of the first user message"},
  {"role": "assistant", "content": "First assistant response"}
]
```

The script:

- Keeps the first user message and first assistant answer  
- Concatenates them as  
  `"user_text [RESPONSE] assistant_text"`  

Default paths:

- `data_file = "sample.jsonl"`  
- `output_dir = "./fine_tuned_model"`  

#### Steps

1. Prepare `sample.jsonl` in the repository root  
2. Run:

```bash
python gpt3.py
```

The script will:

- Load the GPT-2 tokenizer and add special tokens `[PAD]` and `[RESPONSE]`  
- Train with `TrainingArguments` and split data into train and test  
- Save the fine tuned model and tokenizer into `./fine_tuned_model`  

This checkpoint is used in `running_finetuned_model.py` for inference.

### 3. LoRA training on Llama 3 point 2 billion (`train_llama3b.py`)

File: [`train_llama3b.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/train_llama3b.py)

This script is similar to `training_model.py` but uses:

- `LlamaForCausalLM`  
- `AutoTokenizer`  
- Base model `meta-llama/Llama-3.2-1B`  

The data loading pattern is the same as in `training_model.py`:

- Reads JSONL files from `DATA_DIR`  
- Uses `input` and `output` fields and formats them as user and assistant messages  

It then:

1. Builds a LoRA configuration that targets layers such as `q_proj`, `v_proj`, `k_proj`, `c_fc`  
2. Trains with `Trainer` and `DataCollatorForLanguageModeling`  
3. Saves LoRA weights and an optional merged model  

You must have access to the `meta-llama/Llama-3.2-1B` model in your Hugging Face setup.

Run:

```bash
python train_llama3b.py
```

Adjust the paths in the script or through environment variables to match your machine.

## Running a Fine Tuned Model

File: [`running_finetuned_model.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/running_finetuned_model.py)

This small script shows how to:

1. Load a fine tuned GPT-2 model from `./fine_tuned_model`  
2. Tokenize a prompt about AI bias  
3. Generate a continuation  

The script:

- Sets `pad_token` to `eos_token` for the tokenizer  
- Uses `max_new_tokens` to generate a controlled length answer  
- Prints the generated text to the console  

You can change the `prompt` string to test different bias related questions.

Run:

```bash
python running_finetuned_model.py
```

after you have a trained model in `./fine_tuned_model`.

## Bias Evaluation and Metrics

The evaluation part supports several metrics that the paper describes:

- Disparity Index or DI  
- Idea Consistency Score or ICS  
- Thematic Consistency Score or TCS  
- Zero Shot Classification based measures  
- Statistical tests on reported tables  

### 1. Idea Consistency Score with `ics_testing.py`

File: [`ics_testing.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/ics_testing.py)

This script:

1. Loads a fine tuned model and tokenizer from `model_llama_final/lora_weights` or from a path you choose  
2. Uses a SentenceTransformer model such as `paraphrase-MiniLM-L6-v2` to compute similarity between generated responses and ideal unbiased responses  
3. Defines a set of factors and questions, for example  
   - Gender  
   - Race  
   - Socioeconomic status  
   - Disability  
   - Nationality  
   - Religion  
   - Age  
   - Learning style  
   - Language  

4. For each factor, it  
   - Generates a response from the language model  
   - Computes cosine similarity with the expected response  
   - Converts similarity to a score out of 5 and a percentage  

5. Collects results in a pandas DataFrame and plots a bar chart of percentage scores for each factor.

You can run it as:

```bash
python ics_testing.py
```

Make sure the model path and device configuration match your setup.

### 2. Zero shot testing and explainable evaluations with `zero_shot_testing.py`

File: [`zero_shot_testing.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/zero_shot_testing.py)

This script integrates:

- Hugging Face zero shot classification pipeline with `facebook/bart-large-mnli`  
- OpenAI API configured with `OPENAI_API_KEY`  

It defines:

- Evaluation labels such as  
  - Gender bias  
  - Racial bias  
  - Socioeconomic bias  
  - Age bias  
  - Stereotypical behavior  
  - Positive portrayal  
  - Negative portrayal  

- Explainable prompts to analyze model outputs for thematic consistency and bias.

The intended flow is:

1. Load the classifier and OpenAI API key with `dotenv`  
2. Pass candidate texts through the zero shot classifier with the bias labels  
3. Compute label probabilities and build statistics or heatmaps  
4. Optionally ask an OpenAI model for qualitative or explanation based summaries of bias.

Before running:

```bash
export OPENAI_API_KEY="your_api_key_here"
python zero_shot_testing.py
```

You can adapt this script to your own text samples or prompts.

### 3. Statistical tests on reported tables with `t_test.py`

File: [`t_test.py`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/t_test.py)

This script uses:

- `pandas`  
- `scipy.stats.ttest_ind`  
- `numpy`  

It contains hard coded values for:

- English ZSC results from Table 3  
- Hindi ZSC results from Table 4  
- Thematic consistency metrics from Table 5  

Since raw sample level results are not available, the script:

1. Simulates distributions around the reported means  
2. Runs t tests between groups such as male and female or English and Hindi  
3. Prints p values along with the reconstructed tables  

Use it to get a rough sense of statistical significance for differences in reported metrics.

### 4. Notebook for combined analysis with `Metrics_Eval.ipynb`

File: [`Metrics_Eval.ipynb`](https://github.com/tejanshsachdeva/Biasness-Mitigation/blob/master/Metrics_Eval.ipynb)

The notebook is the main place to:

- Load model outputs and evaluation data  
- Compute  
  - Disparity Index or DI  
  - Idea Consistency Score or ICS  
  - Thematic Consistency Score or TCS  
  - Additional implicit or thematic scores  
  - Zero shot based metrics  

- Create plots and tables similar to the figures in the paper  

Open it in Jupyter or VS Code and run cells one by one.

## Configuration and Environment Variables

Several scripts rely on environment variables loaded with `python-dotenv` or `os.getenv`.

Common variables:

- `BASE_DIR`  
  - Used by `training_model.py` and `train_llama3b.py` for data and output locations  

- `MODEL_NAME`  
  - Used to choose the base model in GPT-2 training  

- `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`  
  - LoRA configuration parameters  

- `BATCH_SIZE`, `GRAD_ACCUMULATION`, `LEARNING_RATE`, `NUM_EPOCHS`, `MAX_LENGTH`  
  - Training hyperparameters  

- `OPENAI_API_KEY`  
  - Required for `zero_shot_testing.py`  

You can put these in a `.env` file or export them in your shell.

## Reproducing Results at a High Level

The exact experimental setup in the paper can be complex. At a high level, you can follow this order.

1. Baseline bias evaluation  
   - Use a general purpose model such as GPT-3 point 5 or a standard open source model  
   - Collect outputs for your Hindi and English prompt sets  
   - Process these outputs with  
     - `Metrics_Eval.ipynb`  
     - `zero_shot_testing.py`  
     - `t_test.py`  

2. Prompt engineering  
   - Design prompts that nudge the model toward fair and neutral answers  
   - Run the evaluations again and compare DI, ICS, TCS and ZSC metrics

3. LoRA fine tuning  
   - Build a fairness oriented dataset with `custom_dataset_templates.py` and other data sources  
   - Train a LoRA model with `training_model.py` or `train_llama3b.py`  
   - Run the evaluation scripts again to measure bias reduction

4. Qualitative inspection  
   - Use `running_finetuned_model.py` and custom prompts  
   - Manually inspect differences in tone, portrayal and stereotypes

You can then align your results with the tables and figures from the paper.

## Contributors

- **Tejansh Sachdeva**
- **Mitaali Singhal**
- **Sonia Khetarpaul**

If you use this codebase or build on the ideas, please consider citing the preprint and acknowledging the authors.
