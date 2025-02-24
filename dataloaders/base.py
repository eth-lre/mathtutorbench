from abc import ABC, abstractmethod
from typing import Dict, List, Any
from datasets import load_dataset


class DatasetLoader(ABC):
    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Load dataset and return list of examples"""
        pass


class HuggingFaceDataset(DatasetLoader):
    def __init__(self, dataset_path: str, dataset_name: str, split: str = "test"):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split

    def load(self) -> List[Dict[str, Any]]:
        print(f"Loading dataset {self.dataset_path} part {self.dataset_name}...")
        dataset = load_dataset(self.dataset_path, self.dataset_name, split=self.split)
        return [dict(example) for example in dataset]


class LocalDataset(DatasetLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Dict[str, Any]]:
        import json
        with open(self.file_path, 'r') as f:
            return json.load(f)