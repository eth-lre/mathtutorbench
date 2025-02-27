from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from typing import Dict, List, Union
import json
import os
from tqdm import tqdm
from transformers import set_seed
import logging
from datasets import load_dataset, Dataset
import argparse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = ("Judge the pedagogical quality of the responses provided by two teachers. Focus on the quality of the "
                 "scaffolding guidance, correctness, and actionability of the feedback through nudges, questions "
                 "and hints. Do not give high scores for revealing the full answer.")


def disable_dropout_in_model(model):
    """Disables dropout in the model for inference."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    return model


class PreferenceDataLoader:
    def __init__(self, data_path: str, tokenizer: AutoTokenizer):
        """
        Initialize preference data loader.
        Args:
            data_path: Path to the dataset file
            tokenizer: Tokenizer for processing text
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.raw_data = self._load_raw_data()
        self.dataset = self._format_dataset()

    def _load_raw_data(self):
        """Load raw JSON data."""
        logger.info(f"Loading dataset from {self.data_path}")
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def _format_conversation(self, item, response: str) -> list:
        """Format a conversation with dialog history and response."""
        conversation = []

        # Add system prompt
        system_prompt = {"role": "system",
                         "content": SYSTEM_PROMPT}
        conversation.append(system_prompt)
        conversation.append({"role": "user",
                             "content": "Problem: " + item.get("problem", "") + "\nReference Solution: " + item.get(
                                 "reference_solution", "")})

        # Add the dialog history
        for entry in item["dialog_history"]:
            role = "assistant" if entry["user"] in ["Teacher", "Tutor"] else "user"
            conversation.append({"role": role, "content": entry["text"]})

        # Add the final response
        conversation.append({"role": "assistant", "content": response})
        return conversation

    def _format_dataset(self):
        """Format the dataset into chosen/rejected pairs."""
        formatted_data = {
            'chosen': [],
            'rejected': []
        }

        for item in self.raw_data:
            # Format conversations for chosen (generated) and rejected (ground truth)
            chosen_conv = self._format_conversation(
                item,
                item['generated_teacher_utterance']
            )
            rejected_conv = self._format_conversation(
                item,
                item['ground_truth_response']['text']
            )

            formatted_data['chosen'].append(chosen_conv)
            formatted_data['rejected'].append(rejected_conv)

        # print one example
        print(formatted_data['chosen'][10])
        print(formatted_data['rejected'][10])

        return Dataset.from_dict(formatted_data)  # .shuffle(seed=42)

    def get_evaluation_pairs(self, batch_size: int = None):
        """Get evaluation pairs with optional batching."""
        if batch_size:
            return self.dataset.iter(batch_size=batch_size)
        return self.dataset


class RewardModel:
    def __init__(self, model_name: str):
        """Initialize reward model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            num_labels=1
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = disable_dropout_in_model(self.model).eval()

    @torch.no_grad()
    def get_scores(self, conversations: List[List[dict]], **kwargs) -> List[float]:
        """Get reward scores for a batch of conversations."""
        scores = []
        for conversation in conversations:
            inputs = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(inputs)
            score = outputs.logits[0][0].item()
            scores.append(score)
        return scores


def evaluate_preference_accuracy(
        model_name: str,
        data_path: str,
        batch_size: int = 8,
        output_dir: str = "scaffolding_scores"
) -> Dict[str, Union[float, int]]:
    """Evaluate preference prediction accuracy."""
    model = RewardModel(model_name)
    data_loader = PreferenceDataLoader(data_path, model.tokenizer)

    total_correct = 0
    total_samples = 0
    all_scores = {'chosen': [], 'rejected': []}

    # Copy original data for enrichment
    enriched_data = data_loader.raw_data
    current_idx = 0

    model.model.eval()

    for batch in tqdm(data_loader.get_evaluation_pairs(batch_size), desc="Evaluating"):
        chosen_scores = model.get_scores(batch['chosen'], batch_size=batch_size)
        rejected_scores = model.get_scores(batch['rejected'], batch_size=batch_size)

        # Calculate accuracy
        batch_correct = sum(1 for c, r in zip(chosen_scores, rejected_scores) if c > r)
        total_correct += batch_correct
        total_samples += len(chosen_scores)

        # Store scores
        all_scores['chosen'].extend(chosen_scores)
        all_scores['rejected'].extend(rejected_scores)

        # Add scores to the enriched data
        for c_score, r_score in zip(chosen_scores, rejected_scores):
            enriched_data[current_idx]['chosen_score'] = float(c_score)
            enriched_data[current_idx]['rejected_score'] = float(r_score)
            current_idx += 1

        logger.info(f"Current accuracy: {total_correct / total_samples:.4f}")

    # Calculate final metrics
    accuracy = total_correct / total_samples
    results = {
        'win_rate': accuracy,
        'score': float(np.mean(all_scores['chosen'])),
        'baseline_score': float(np.mean(all_scores['rejected'])),
        'mean_margin': float(np.mean(np.array(all_scores['chosen']) - np.array(all_scores['rejected']))),
        'total_samples': total_samples,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base_name = data_path.split('/')[-1]

    # Save metrics
    metrics_path = os.path.join(output_dir, f"{base_name}_results.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save enriched data in same format as input
    enriched_path = os.path.join(output_dir, f"{base_name}_enriched_data.json")
    with open(enriched_path, 'w') as f:
        json.dump(enriched_data, f, indent=2)

    # Update results yaml file
    path = Path(args.data_path)
    parts = path.stem.split('-')  # Split filename by '-'
    model_name = parts[1]  # Extract model name
    task_config_name = parts[2]  # Extract task config name

    results_yaml_file = "../results/" + f"results-{model_name}.yaml"
    with open(results_yaml_file, 'r') as f:
        data = results_yaml_file.safe_load(f)

    if not data:
        data = {}
    # Find the matching task config and update
    if task_config_name in data:
        data[task_config_name]['results'] = results
    else:
        data[task_config_name] = {'results': results}

    with open(results_yaml_file, 'w') as f:
        yaml.dump(data, f)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    set_seed(42)

    parser = argparse.ArgumentParser(description='Run scaffolding score reward model on generations.')
    parser.add_argument('--data_path', type=str, help='Path to the data with generations.')
    args = parser.parse_args()

    MODEL_NAME = "eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel"
    results = evaluate_preference_accuracy(MODEL_NAME, args.data_path)
    print(f"Final Results: {results}")

