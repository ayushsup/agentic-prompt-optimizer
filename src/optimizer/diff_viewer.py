import difflib
import os

class DiffViewer:
    def __init__(self, output_dir: str = "logs/diffs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_diff(self, old_prompt: str, new_prompt: str, iteration: int):
        diff = difflib.unified_diff(
            old_prompt.splitlines(keepends=True),
            new_prompt.splitlines(keepends=True),
            fromfile=f'Prompt_v{iteration-1}',
            tofile=f'Prompt_v{iteration}',
            n=3
        )
        
        diff_text = "".join(diff)
        file_path = os.path.join(self.output_dir, f"diff_iteration_{iteration}.diff")
        
        with open(file_path, "w") as f:
            f.write(diff_text)
            
        return diff_text