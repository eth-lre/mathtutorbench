from typing import Dict, List, Any

from tasks.extraction import extract_ground_truth_questions
from registry import TaskRegistry
from .base import Task
import sacrebleu
import re
from dataclasses import dataclass


@TaskRegistry.register("socratic_questioning")
class SocraticQuestioningTask(Task):
    def parse_response(self, response: str) -> List[str]:
        """Extract questions from the model's response"""
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            questions.append(line)
        return questions


    def compute_metrics(self, predictions: List[List[str]], targets: List[str]) -> Dict[str, float]:
        """Compute SACREBLEU scores for generated questions"""
        # Extract ground truth questions for each target
        print(targets)
        print(predictions)

        target_questions = [extract_ground_truth_questions(target) for target in targets]

        print(target_questions)
        print(predictions)

        # Compute SACREBLEU scores for each example
        scores = []
        question_counts = []


        for pred_questions, target_questions_list in zip(predictions, target_questions):
            if not pred_questions or not target_questions_list:
                continue

            # Create references list in format required by SACREBLEU
            references = [q for q in target_questions_list]

            # Calculate SACREBLEU for each predicted question against all reference questions
            question_scores = []
            for pred in pred_questions:
                bleu = sacrebleu.sentence_bleu(pred, references)
                question_scores.append(bleu.score)

            if question_scores:
                scores.append(max(question_scores))  # Use best match for each prediction
                question_counts.append(min(len(pred_questions), len(target_questions_list)))

        avg_bleu = sum(scores) / len(scores) if scores else 0.0
        avg_questions = sum(question_counts) / len(question_counts) if question_counts else 0.0

        return {
            "bleu": avg_bleu / 100.0,  # Normalize to 0-1 range
            "avg_questions": avg_questions
        }
