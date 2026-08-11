# MathTutorBench: A Benchmark for Measuring Open-ended Pedagogical Capabilities of LLM Tutors
[![Arxiv](https://img.shields.io/badge/Arxiv-2502.18940-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.18940)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/deed.en)
[![Python Versions](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

## Overview
**MathTutorBench** is a benchmark which provides a unified framework for evaluating open-ended pedagogical capabilities of large langauge models (LLMs) tutors across three high level teacher skills and seven concrete tasks.


## Key Features
- **Automatic Evaluation**: The benchmark is designed to be run automatically on any new models you are developing.
- **Comprehensive Metrics**: The benchmark covers a three high level tasks skills and seven tasks to evaluate in the domain of math tutoring.
- **Teacher-Grounded Evaluation**: Each task is annotated with teacher ground truths and compared to it.
- **Fast execution loop**: Run benchmark on different tasks very quickly.

<p align="center">
<img src="./images/skills.png" alt="Skills" width="400">
</p>

## Quick Start - Evaluate a New Model
### 0. Run your model locally using vllm - skip if you are using API
For more details on how to run your model locally using vllm, see [vllm](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#vllm-server) documentation. Optionally add tensor parallelism if you have multiple GPUs and your model is large.
```bash
vllm serve [[model_name]] --seed 42 --tensor-parallel-size 4
```

### 1. Run task(s) from the benchmark
```bash
# Example with vllm model
python main.py --tasks mistake_location.yaml --provider completion_api --model_args base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct
# Example with OpenAI API
python main.py --tasks mistake_correction.yaml --provider completion_api --model_args model=gpt-4o-mini-2024-07-18,api_key=<API_KEY>
# Example with LearnLM Gemini API
python main.py --tasks student_solution_correctness.yaml --provider gemini --model_args model==learnlm-1.5-pro-experimental,api_key=<API_KEY>

```
- Required:
  - `--tasks`: Task definition file in the `configs` folder. Use comma `,` separated list for multiple sequential tasks.
    - `problem_solving.yaml`: Task definition for problem solving.
    - `socratic_questioning.yaml`: Task definition for socratic questioning.
    - `student_solution_correctness.yaml`: Task definition for student solution generation.
    - `mistake_location.yaml`: Task definition for mistake location.
    - `mistake_correction.yaml`: Task definition for mistake correction.
    - `scaffolding_generation.yaml`: Task definition for scaffolding generation.
    - `pedagogy_following.yaml`: Task definition for pedagogy following.
    - `scaffolding_generation_hard.yaml`: Task definition for scaffolding generation hard.
    - `pedagogy_following_hard.yaml`: Task definition for pedagogy following hard.
  - `--provider`: API provider to use for the task.
    - `completion_api`: Use the completion API for the task. Support any OpenAI-type API. Use for openai and vllm models.
    - `gemini`: Use the gemini API for the task. 
  - `--model_args`: Model arguments to pass to the API provider.
    - `base_url`: Base URL of the API provider. Empty for openai and gemini.
    - `model`: Model name to use for the task. Default is the first available model.
    - `api_key`: API key to access API. Empty for vllm models.
    - `is_chat`: Whether the requests to the model should use chat-based template (Chat Completion API). Default is False.
    - `is_thinking`: Set to True for thinking/reasoning models (e.g. Qwen3) to disable thinking during generation. Default is False. In completion mode, an empty think block (`<think>\n\n</think>`) is appended to the prompt; in chat mode (`is_chat=True`), `chat_template_kwargs={"enable_thinking": False}` is sent to the server (supported by vllm).
    - `temperature`: Temperature for sampling. Default is 0.0.
    - `max_tokens`: Maximum tokens to generate. Default is 2048.
    - `max_retries`: Maximum retries for the API. Default is 3.
  - Optional:
    - `--output`: Output directory for the results. Default is `results`.
    - `--debug`: Run only on a small subset of examples for debugging.

### Examples for running each task
Examples below use a local vllm model. Run the tasks of the Pedagogical Ability skill (`mistake_correction`, `scaffolding_generation`, `pedagogy_following`, `scaffolding_generation_hard`, `pedagogy_following_hard`) with `is_chat=True` by default, as they are dialog-based.
```bash
python main.py --tasks problem_solving.yaml --provider completion_api --model_args base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct
python main.py --tasks socratic_questioning.yaml --provider completion_api --model_args base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct
python main.py --tasks student_solution_correctness.yaml --provider completion_api --model_args base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct
python main.py --tasks mistake_location.yaml --provider completion_api --model_args base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct
python main.py --tasks mistake_correction.yaml --provider completion_api --model_args base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct,is_chat=True
python main.py --tasks scaffolding_generation.yaml --provider completion_api --model_args base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct,is_chat=True
python main.py --tasks pedagogy_following.yaml --provider completion_api --model_args base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct,is_chat=True
python main.py --tasks scaffolding_generation_hard.yaml --provider completion_api --model_args base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct,is_chat=True
python main.py --tasks pedagogy_following_hard.yaml --provider completion_api --model_args base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct,is_chat=True
```

### Using OpenRouter
[OpenRouter](https://openrouter.ai/) exposes many models behind an OpenAI-compatible API, so it works with the `completion_api` provider. Set `base_url` to the OpenRouter endpoint, pass your OpenRouter API key, and use the full model name from OpenRouter (e.g. `anthropic/claude-sonnet-4.6`):
```bash
python main.py --tasks problem_solving.yaml --provider completion_api --model_args base_url=https://openrouter.ai/api/v1/,api_key=sk-XX,model=anthropic/claude-sonnet-4.6 --output results/ --debug
```
Remove `--debug` to run on the full dataset.

> [!NOTE]
> **Cost estimate**: Running the full benchmark via OpenRouter costs around $35 USD for non-reasoning models (e.g. Claude Sonnet 4.6) and can reach $100–200 USD for reasoning models due to the additional reasoning tokens. Use `--debug` first to verify your setup on a small subset before a full run.

The performance of different benchmarked models averaged across tasks for Qwen2.5 family is as follows (using vllm version 0.8.0 on one node with 4x GH200 GPUs):

| Model                  | Total time [min] | Examples/sec | Tokens/sec |
|-------------------------|------------------|--------------|------------|
| Qwen2.5-1.5B-Instruct  | 61.1             | 2.73         | 757.6      |
| Qwen2.5-7B-Instruct    | 58.3             | 2.86         | 1012       |
| Qwen2.5-32B-Instruct   | 545.3            | 0.31         | 166.3      |
| Qwen2.5-72B-Instruct   | 233.9            | 0.71         | 135.2      |


### 2. Run reward model of the Pedagogical Ability tasks
Set the `--data_path` to model outputs of the pedagogical ability tasks. The model computes win rates of generated teacher utterance over the ground truth teacher utterance.
```bash
python reward_model/compute_scaffolding_score.py --data_path results/generations-<specific-model>.json
```

As the model is small in size (1.5B parameters), running the full evaluation should be fast (within 10 minutes on a single GPU).
Reward model computation performance with different batch sizes on a single GH200 GPU:

| Batch size | Total time [sec] | Examples/sec | Tokens/sec |
|------------|------------------|--------------|------------|
| 1          | 419.58           | 7.01         | 6928.0     |
| 8          | 406.08           | 7.25         | 7159.3     |
| 64         | 413.28           | 7.12         | 7034.8     |
| 128        | 408.87           | 7.20         | 7110.0     |



### 3. Visualize results
Results are available in the `results` folder. To visualize the results, run:
```bash
python visualize.py --results_dir results/
```

<img src="./images/radar_chart.png" alt="Radar chart comparing model performance across benchmark tasks" width="800">


## Installation
```bash
pip install -r requirements.txt
```

## Leaderboard
| Model | Problem Solving | Socratic Questioning | Solution Correctness | Mistake Location | Mistake Correction | Scaffolding Win Rate | Pedagogy IF Win Rate | Scaffolding (Hard) | Pedagogy IF (Hard) |
|--------|----------------|----------------------|----------------------|------------------|-------------------|------------------|-----------------|----------------|------------------|
| LLaMA3.2-3B-Instruct | 0.60 | 0.29 | 0.67 | 0.41 | 0.13 | 0.64 | 0.63 | 0.45 | 0.40 |
| LLaMA3.1-8B-Instruct | 0.70 | 0.29 | 0.63 | 0.29 | 0.09 | 0.61 | 0.67 | 0.46 | 0.49 |
| LLaMA3.1-70B-Instruct | 0.91 | 0.29 | 0.71 | 0.56 | 0.19 | 0.63 | 0.70 | 0.49 | 0.49 |
| GPT-4o | 0.90 | **0.48** | 0.67 | 0.37 | 0.84 | 0.50 | 0.82 | 0.46 | 0.70 |
| LearnLM-1.5-Pro | 0.94 | 0.32 | 0.75 | 0.57 | 0.74 | 0.64 | 0.68 | 0.66 | 0.67 |
| Llemma-7B-ScienceTutor | 0.62 | 0.29 | 0.66 | 0.29 | 0.16 | 0.37 | 0.48 | 0.38 | 0.42 |
| Qwen2.5-7B-SocraticLM | 0.73 | 0.32 | 0.05 | 0.39 | 0.23 | 0.39 | 0.39 | 0.28 | 0.28 |
| Qwen2.5-Math-7B-Instruct | 0.88 | 0.35 | 0.43 | 0.47 | 0.49 | 0.06 | 0.07 | 0.05 | 0.05 |
| apertus-ai/Apertus-v1.5-8B | 0.76 | 0.27 | 0.35 | 0.46 | 0.07 | 0.11 | 0.22 | 0.09 | 0.12 |
| zai-org/GLM-4.7-Flash | 0.78 | 0.29 | 0.59 | 0.41 | 0.09 | 0.39 | 0.70 | 0.30 | 0.57 |
| Qwen/Qwen3.6-27B | 0.97 | 0.28 | 0.80 | 0.71 | 0.89 | 0.57 | 0.78 | 0.62 | 0.78 |
| google/gemini-2.5-pro (adaptive reasoning) | 0.88 | 0.30 | 0.86 | 0.72 | 0.59 | 0.67 | 0.53 | 0.69 | 0.67 |
| eth-nlped/TutorRL-7B | 0.77 | 0.23 | 0.65 | 0.36 | 0.75 | 0.53 | 0.70 | 0.55 | 0.66 |
| anthropic/claude-sonnet-4.6 | 0.91 | 0.32 | 0.80 | 0.55 | 0.79 | 0.49 | **0.85** | 0.53 | **0.80** |
| google/gemini-3.1-pro (default adaptive reasoning) | 0.97 | 0.36 | **0.89** | **0.81** | **0.93** | **0.73** | 0.73 | **0.76** | 0.73 |
| google/gemini-3.6-flash (default adaptive reasoning) | **0.98** | 0.34 | 0.88 | 0.80 | 0.91 | 0.69 | 0.72 | 0.69 | 0.76 |
| Qwen/Qwen2.5-7B-Instruct | 0.85 | 0.19 | 0.64 | 0.49 | 0.52 | 0.44 | 0.40 | 0.42 | 0.44 |

> [!NOTE]
> All models are evaluated with thinking/reasoning **disabled** (see the `is_thinking` model argument). The only exceptions are the `google/gemini-2.5-pro`, `google/gemini-3.1-pro` and `google/gemini-3.6-flash` models, which use their default adaptive thinking (thinking budget set to `-1`).

Averaging the subtasks into the three high-level teacher skills — Math Expertise (problem solving, socratic questioning), Student Understanding (solution correctness, mistake location, mistake correction) and Pedagogy (the four win rates) — shows that strong problem solvers are not automatically strong tutors:

<img src="./images/skills_scatter.png" alt="Scatter of Student Understanding vs Pedagogy, colored by Math Expertise" width="700">


## Submit your model to leaderboard
To submit your model to the leaderboard, please follow the steps below:
1. Open a new issue with the title `Leaderboard Submission: <Model Name>`.
2. Provide the exact model name on the Huggingface hub and if specific code/arguments/settings are needed for the model or the vllm library which will be used to run your model. Please copy the results from the local run of the model.

## Adding a New Task
Please open a new PR and provide the configuration of the task in the `configs` folder and the task implementation in the `tasks` folder.

# Scaffolding Score Pedagogical Reward Model
- [Dataset](https://huggingface.co/datasets/dmacjam/pedagogical-rewardmodel-data) used to train and evaluate the Scaffolding score reward model

## Citation
Please cite as:
```bibtex
@inproceedings{macina-etal-2025-mathtutorbench,
    title = "{M}ath{T}utor{B}ench: A Benchmark for Measuring Open-ended Pedagogical Capabilities of {LLM} Tutors",
    author = "Macina, Jakub  and
      Daheim, Nico  and
      Hakimi, Ido  and
      Kapur, Manu  and
      Gurevych, Iryna  and
      Sachan, Mrinmaya",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.11/",
    doi = "10.18653/v1/2025.emnlp-main.11",
    pages = "204--221",
    ISBN = "979-8-89176-332-6",
}
```

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
