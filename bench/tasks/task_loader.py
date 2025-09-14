"""Task loader for Ko-AgentBench tasks."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class TaskLoader:
    """Loads and manages benchmark tasks from JSONL files."""
    
    def __init__(self, tasks_dir: Optional[Path] = None):
        """Initialize task loader.
        
        Args:
            tasks_dir: Directory containing task JSONL files
        """
        self.tasks_dir = tasks_dir or Path(__file__).parent
        self.tasks: List[Dict[str, Any]] = []
    
    def load_tasks(self, file_path: str) -> List[Dict[str, Any]]:
        """Load tasks from JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        with open(self.tasks_dir / file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    task = json.loads(line.strip())
                    tasks.append(task)
        return tasks
    
    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task dictionary or None if not found
        """
        for task in self.tasks:
            if task.get('id') == task_id:
                return task
        return None
    
    def filter_tasks(self, category: Optional[str] = None, 
                    difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter tasks by category and difficulty.
        
        Args:
            category: Task category filter
            difficulty: Task difficulty filter
            
        Returns:
            Filtered list of tasks
        """
        filtered = self.tasks
        
        if category:
            filtered = [t for t in filtered if t.get('category') == category]
        
        if difficulty:
            filtered = [t for t in filtered if t.get('difficulty') == difficulty]
            
        return filtered