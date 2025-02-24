import argparse
import json
from pathlib import Path
import yaml
from typing import Dict, Any

from completion_api import create_llm_model, LLMConfig
from tasks.base import TaskConfig
from registry import TaskRegistry
from tqdm import tqdm


def parse_model_args(args_str: str) -> Dict[str, Any]:
    """Parse comma-separated key=value pairs into a dictionary"""
    if not args_str:
        return {}

    args_dict = {}
    for pair in args_str.split(','):
        key, value = pair.split('=')
        # Convert string values to appropriate types
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)
        args_dict[key] = value
    return args_dict


def load_task_config(config_path: str) -> TaskConfig:
    with open("configs/" + config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return TaskConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(description='Run model evaluation on educational tasks')
    parser.add_argument('--tasks', type=str, required=True,
                        help='Comma-separated list of task config YAML files')
    parser.add_argument("--provider", required=False, choices=['completion_api', 'ollama', 'gemini'],
                        default='ollama', help="LLM provider to use")
    parser.add_argument('--model_args', type=str, required=True,
                        help='Model arguments in key=value format')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    args = parser.parse_args()

    # Parse model arguments
    model_args = parse_model_args(args.model_args)
    model_args['provider'] = args.provider

    # Create config and model
    config = LLMConfig(**model_args)
    model = create_llm_model(config)


    # Process each task
    task_paths = [path.strip() for path in args.tasks.split(',')]
    results = {}

    for task_path in task_paths:
        # Load task configuration
        task_config = load_task_config(task_path)
        print(task_path)

        # Get task class and initialize
        task_cls = TaskRegistry.get_task(task_config.name)
        task = task_cls(task_config)

        predictions = []
        targets = []
        all_generations = []

        # Process examples
        for example in tqdm(task.get_test_examples(), desc=f"Evaluating {task_config.name}"):
            # Prepare messages with few-shot examples if provided
            messages = []
            example["shots"] = task_config.few_shot_samples
            # Get model response
            response = model.generate(
                messages=messages,
                system_prompt=task.get_system_prompt(example),
                stop=task_config.stop
            )
            # Parse and store prediction
            prediction = task.parse_response(response)
            predictions.append(prediction)
            print(prediction)
            formatted_ground_truth = task.format_ground_truth(example)
            print(formatted_ground_truth)
            targets.append(formatted_ground_truth)

            if "pedagogy" in task_config.name:
                generation = {
                    "problem": example.get("question", ""),
                    "reference_solution": example.get("reference_solution", "N/A"),
                    "dialog_history": example.get("conversation_json", []),
                    "dialog_formatted": example.get("dialog_history", ""),
                    "ground_truth_response": example.get("ground_truth_response", ""),
                    "generated_teacher_utterance": prediction,
                }
                all_generations.append(generation)

        # Compute metrics
        metrics = task.compute_metrics(predictions, targets)
        results[task_config.name] = metrics

    print(results)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"results-{config.model.split('/')[-1]}.yaml", 'a+') as f:
        yaml.dump(results, f)

    if len(all_generations) > 0:
        output_file = output_dir / f"generations-{config.model.split('/')[-1]}-{task_config.name}.json"
        with open(output_file, 'w') as f:
            json.dump(all_generations, f, indent=2)


if __name__ == "__main__":
    main()