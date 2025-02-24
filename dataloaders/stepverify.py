import json
from typing import List, Dict, Any
from .base import DatasetLoader


class StepVerifyDataset(DatasetLoader):
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
            cutoff = len(conversation_list)

            # Find the cutoff point where the student response starts
            for i, turn in enumerate(conversation_list):
                if turn["user"] == "Student":
                    cutoff = i
                    break
                formatted_dialog.append(f"Teacher: {turn['text']}")

            dialog_history_str = "\n".join(formatted_dialog)

            # Create example with error
            error_example = {
                'question': example['problem'],
                'student_solution': "\\n".join(["Step " + str(sub_index + 1) + " - " + substep for sub_index, substep in
                                               enumerate(example["student_incorrect_solution"][:-1])]),
                'is_error': True,
                'error_step': int(example['incorrect_index']) + 1,  # Convert to 1-based indexing
                'dialog_history': dialog_history_str,
                "student_chat_solution": conversation_list[cutoff]['text'],
                "reference_solution": example["reference_solution"],
            }
            processed_examples.append(error_example)

            # Create example without error (using reference solution)
            no_error_example = {
                'question': example['problem'],
                'student_solution': "\\n".join(["Step " + str(sub_index + 1) + " - " + substep for sub_index, substep in
                                               enumerate(example["reference_solution"].split("\n")[:-1])]),
                'is_error': False,
                'error_step': 0,  # No error
                'dialog_history': dialog_history_str,
                "student_chat_solution": example["student_correct_response"],
                "reference_solution": example["reference_solution"],
            }
            processed_examples.append(no_error_example)

        return processed_examples
