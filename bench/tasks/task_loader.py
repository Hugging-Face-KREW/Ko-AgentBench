"""Task loader for Ko-AgentBench tasks."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - optional dependency fallback
    Draft202012Validator = None


class TaskLoader:
    """Loads and manages benchmark tasks from JSON/JSONL files."""
    
    def __init__(
        self,
        tasks_dir: Optional[Path] = None,
        schema_dir: Optional[Path] = None,
    ):
        """Initialize task loader.
        
        Args:
            tasks_dir: Directory containing task JSON/JSONL files
            schema_dir: Directory containing v2 JSON schemas
        """
        self.tasks_dir = Path(tasks_dir) if tasks_dir is not None else Path(__file__).parent
        self.schema_dir = Path(schema_dir) if schema_dir is not None else (self.tasks_dir.parent / "schemas" / "v2")
        self.tasks: List[Dict[str, Any]] = []
        self._v2_task_validator = self._load_v2_task_validator()

    def _load_v2_task_validator(self):
        """Load Draft 2020-12 validator for v2 task schema if available."""
        if Draft202012Validator is None:
            return None

        schema_path = self.schema_dir / "task.schema.json"
        if not schema_path.exists():
            return None

        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            return Draft202012Validator(schema)
        except Exception:
            return None

    @staticmethod
    def _is_v2_task(task: Dict[str, Any]) -> bool:
        """Return True when task looks like a v2 task."""
        schema_version = task.get("schema_version")
        return isinstance(schema_version, str) and schema_version.startswith("2.")
    
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

    def validate_tasks(
        self,
        tasks: List[Dict[str, Any]],
        *,
        file_label: str = "",
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Validate v2 tasks against JSON schema.

        Args:
            tasks: Raw task list loaded from JSON/JSONL.
            file_label: Optional filename label for reporting.
            strict: If True, raise ValueError on validation errors.

        Returns:
            Validation summary dictionary.
        """
        summary = {
            "file": file_label,
            "total_tasks": len(tasks),
            "v2_tasks": 0,
            "valid_v2_tasks": 0,
            "errors": [],
            "validator_available": self._v2_task_validator is not None,
        }

        if not tasks:
            return summary

        for idx, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            if not self._is_v2_task(task):
                continue

            summary["v2_tasks"] += 1

            if self._v2_task_validator is None:
                # Schema validator unavailable: only count v2 task.
                summary["valid_v2_tasks"] += 1
                continue

            errors = sorted(
                self._v2_task_validator.iter_errors(task),
                key=lambda e: list(e.path),
            )
            if not errors:
                summary["valid_v2_tasks"] += 1
                continue

            error_messages = []
            for e in errors:
                path = ".".join(str(p) for p in e.path) or "<root>"
                error_messages.append(f"{path}: {e.message}")

            summary["errors"].append(
                {
                    "index": idx,
                    "task_id": task.get("task_id"),
                    "messages": error_messages,
                }
            )

        if strict and summary["errors"]:
            first = summary["errors"][0]
            err = "; ".join(first.get("messages", [])[:3])
            raise ValueError(
                f"v2 task schema validation failed ({file_label}): "
                f"task_id={first.get('task_id')} - {err}"
            )
        if strict and summary["v2_tasks"] > 0 and not summary["validator_available"]:
            raise ValueError(
                f"v2 task schema validation unavailable ({file_label}): "
                "jsonschema validator is not available."
            )

        return summary
    
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
