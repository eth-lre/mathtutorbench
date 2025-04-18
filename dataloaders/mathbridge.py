import json
from typing import List, Dict, Any
from dataloaders.base import DatasetLoader


class MathBridge(DatasetLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Dict[str, Any]]:
        with open(self.file_path, 'r') as f:
            raw_data = json.load(f)

        processed_examples = []
        for example in raw_data:
            # Process dialog history
            conversation_list = example.get("dialog_history", [])
            formatted_dialog = []
            cutoff = len(conversation_list) - 1

            # Find the cutoff point where the student response starts
            for i, turn in enumerate(conversation_list):
                if turn["user"] == "Student":
                    formatted_dialog.append(f"Student: {turn['text']}")
                else:
                    formatted_dialog.append(f"Teacher: {turn['text']}")

            # Remove last teacher turn
            if len(formatted_dialog) > 1:
                formatted_dialog = formatted_dialog[:-1]
            dialog_history_str = "\n".join(formatted_dialog)

            # Create example with error
            error_example = {
                'question': example['problem'],
                'conversation_json': example.get("dialog_history", [])[:cutoff],
                "ground_truth_response": example.get("dialog_history", [])[cutoff],
                'dialog_history': dialog_history_str,
                "reference_solution": example["reference_solution"],
            }
            processed_examples.append(error_example)

        return processed_examples
