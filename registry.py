from typing import Type, Dict
from tasks.base import Task

class TaskRegistry:
    _tasks: Dict[str, Type[Task]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(task_cls: Type[Task]):
            cls._tasks[name] = task_cls
            return task_cls
        return decorator

    @classmethod
    def get_task(cls, name: str) -> Type[Task]:
        if name not in cls._tasks:
            raise ValueError(f"Unknown task: {name}")
        return cls._tasks[name]