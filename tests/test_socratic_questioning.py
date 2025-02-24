import unittest

from extraction import extract_ground_truth_questions, extract_ground_truth_questions_and_step


class TestExtractGroundTruthQuestions(unittest.TestCase):

    def test_extract_questions(self):
        # Sample input and expected output
        input_answer = "How many eggs does Janet sell? ** Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nHow much does Janet make at the farmers' market? ** She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18"
        expected_output = ['How many eggs does Janet sell?', "How much does Janet make at the farmers' market?"]

        # Calling the function to extract questions
        result = extract_ground_truth_questions(input_answer)

        # Asserting the expected output matches the result
        self.assertListEqual(result, expected_output)

    def test_extract_questions(self):
        # Sample input and expected output
        input_answer = "How many eggs does Janet sell? ** Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nHow much does Janet make at the farmers' market? ** She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18"
        expected_questions_output = ['How many eggs does Janet sell?', "How much does Janet make at the farmers' market?"]
        expected_steps_output = ['Janet sells 16 - 3 - 4 = 9 duck eggs a day.', "She makes 9 * 2 = $18 every day at the farmer’s market."]

        # Calling the function to extract questions
        questions, steps = extract_ground_truth_questions_and_step(input_answer)
        print(questions)
        print(steps)

        # Asserting the expected output matches the result
        self.assertListEqual(questions, expected_questions_output)
        self.assertListEqual(steps, expected_steps_output)
