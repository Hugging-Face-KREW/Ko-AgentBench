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
        """Load tasks from JSONL or JSON file.
        
        Args:
            file_path: Path to JSONL or JSON file
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        file_full_path = self.tasks_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
        
        with open(file_full_path, 'r', encoding='utf-8') as f:
            # Try to load as JSON array first
            content = f.read()
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    tasks = data
                else:
                    tasks = [data]
            except json.JSONDecodeError:
                # If JSON fails, try JSONL format
                for line in content.split('\n'):
                    if line.strip():
                        task = json.loads(line.strip())
                        tasks.append(task)
        
        self.tasks = tasks
        return tasks
    
    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task dictionary or None if not found
        """
        for task in self.tasks:
            if task.get('task_id') == task_id or task.get('id') == task_id:
                return task
        return None
    
    def filter_tasks(self, task_level: Optional[int] = None, task_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter tasks by category and difficulty.
        
        Args:
            task_level: Task level filter 
            task_category: Task category filter 
            
        Returns:
            Filtered list of tasks
        """
        filtered = self.tasks

        if task_category:
            filtered = [t for t in filtered if t.get('task_category') == task_category]
        
        if task_level is not None:
            filtered = [t for t in filtered if t.get('task_level') == task_level]
            
        return filtered