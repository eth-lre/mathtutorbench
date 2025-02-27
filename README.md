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
For more details on how to run your model locally using vllm, see [vllm](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#vllm-server) documentation.
```bash
vllm serve [[model_name]]   
```

### 1. Run task(s) from the benchmark
```bash
# Example with openai API
python main.py --tasks mistake_location.yaml --provider completion_api --model_args model=gpt-4o-mini-2024-07-18,is_chat=True,api_key=<API_KEY>
# Example with vllm model
python main.py --tasks mistake_location.yaml --provider completion_api --model_args base_url=base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct,is_chat=True
```
- Required:
  - `--tasks`: Task definition file in the `configs` folder. Use comma `,` separated list for multiple sequential tasks.
    - `problem_solving.yaml`: Task definition for problem solving.
    - `socratic_questioning.yaml`: Task definition for socratic questioning.
    - `student_solution_generation.yaml`: Task definition for student solution generation.
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
    - `is_chat`: Whether the model is chat-based or not. Default is False.
    - `temperature`: Temperature for sampling. Default is 0.0.
    - `max_tokens`: Maximum tokens to generate. Default is 2048.
    - `max_retries`: Maximum retries for the API. Default is 3.


### 2. Run reward model of the Pedagogical Ability tasks
Set the `--data_path` to model outputs of the pedagogical ability tasks. The model computes win rates of generated teacher utterance over the ground truth teacher utterance.
```bash
python reward_models/compute_scaffolding_score.py --data_path results/generations-<specific-model>.json
```

### 3. Visualize results
```bash
python visualize.py --results_dir results/
```

<img src="./images/figure2.png" alt="Skills" width="800">


## Installation
```bash
pip install -r requirements.txt
```

## Leaderboard
| Model | Problem Solving | Socratic Questioning | Solution Correctness | Mistake Location | Mistake Correction | Scaffolding Win Rate | Pedagogy IF Win Rate | Scaffolding (Hard) | Pedagogy IF (Hard) |
|--------|----------------|----------------------|----------------------|------------------|-------------------|------------------|-----------------|----------------|------------------|
| LLaMA3.2-3B-Instruct | 0.60 | 0.29 | 0.67 | 0.41 | 0.13 | **0.64** | 0.63 | 0.45 | 0.40 |
| LLaMA3.1-8B-Instruct | 0.70 | 0.29 | 0.63 | 0.29 | 0.09 | 0.61 | 0.67 | 0.46 | 0.49 |
| LLaMA3.1-70B-Instruct | 0.91 | 0.29 | 0.71 | 0.56 | 0.19 | 0.63 | 0.70 | 0.49 | 0.49 |
| GPT-4o | 0.90 | **0.48** | 0.67 | 0.37 | **0.84** | 0.50 | **0.82** | 0.46 | **0.70** |
| LearnLM-1.5-Pro | **0.94** | 0.32 | **0.75** | **0.57** | 0.74 | **0.64** | 0.68 | **0.66** | 0.67 |
| Llemma-7B-ScienceTutor | 0.62 | 0.29 | 0.66 | 0.29 | 0.16 | 0.37 | 0.48 | 0.38 | 0.42 |
| Qwen2.5-7B-SocraticLM | 0.73 | 0.32 | 0.05 | 0.39 | 0.23 | 0.39 | 0.39 | 0.28 | 0.28 |
| Qwen2.5-Math-7B-Instruct | 0.88 | 0.35 | 0.43 | 0.47 | 0.49 | 0.06 | 0.07 | 0.05 | 0.05 |


## Adding a New Task
Will be updated soon.

## Submit your model to leaderboard
Will be updated soon.

## Citation
Please cite as:
```bibtex
@article{macina2025mathtutorbench,
      title={MathTutorBench: A Benchmark for Measuring Open-ended\\ Pedagogical Capabilities of LLM Tutors}, 
      author={Jakub Macina and Nico Daheim and Ido Hakimi and Manu Kapur and Iryna Gurevych and Mrinmaya Sachan},
      year={2025},
      eprint={2502.18940},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18940},
}
```
