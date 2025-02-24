from typing import List
import re

def extract_ground_truth_questions(answer: str) -> List[str]:
    """Extract questions from ground truth answer"""
    questions = []
    all_sent = answer.split("\n")
    for single_sent in all_sent:
        if len(single_sent.split(" ** ")) == 2:
            questions.append(single_sent.split(" ** ")[0])
    return questions

def extract_ground_truth_questions_and_step(answer: str) -> List[str]:
    """Extract questions from ground truth answer"""
    questions, steps = [], []
    all_sent = answer.split("\n")
    for single_sent in all_sent:
        if len(single_sent.split(" ** ")) == 2:
            parsed = single_sent.split(" ** ")
            questions.append(parsed[0])
            step = parsed[1]
            # Remove unnecessary calculator tokens
            step = re.sub(r"<<.*?>>", "", step)
            steps.append(step)
    return questions, steps
