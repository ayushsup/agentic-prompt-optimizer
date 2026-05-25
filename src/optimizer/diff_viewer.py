"""
Diff viewer for accepted prompt mutations.

Generates a unified diff between successive accepted prompts and saves
each diff to logs/diffs/diff_iteration_N.diff.

The diff is between the PREVIOUSLY ACCEPTED prompt and the NEWLY ACCEPTED
prompt — it always shows exactly what changed at each acceptance event.
"""

import difflib
import os


class DiffViewer:
    def __init__(self, output_dir: str = "logs/diffs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_diff(
        self,
        old_prompt: str,
        new_prompt: str,
        iteration: int,
        old_label: str | None = None,
        new_label: str | None = None,
    ) -> str:
        """
        Write a unified diff between old_prompt and new_prompt.

        Parameters
        ----------
        old_prompt  : Previously accepted prompt text.
        new_prompt  : Newly accepted prompt text.
        iteration   : Current iteration index (used in filename and labels).
        old_label   : Display label for the old prompt (default: Prompt_vN-1).
        new_label   : Display label for the new prompt (default: Prompt_vN).

        Returns
        -------
        The unified diff string (also written to disk).
        """
        from_label = old_label or f"Prompt_v{max(0, iteration - 1)}"
        to_label   = new_label or f"Prompt_v{iteration}"

        # Ensure consistent line endings for stable diffs
        old_lines = old_prompt.splitlines(keepends=True)
        new_lines = new_prompt.splitlines(keepends=True)

        # Append newline to last line if missing (cleaner diff output)
        if old_lines and not old_lines[-1].endswith("\n"):
            old_lines[-1] += "\n"
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        diff_lines = list(difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=from_label,
            tofile=to_label,
            n=3,
        ))

        diff_text = "".join(diff_lines)

        if not diff_text:
            diff_text = f"# No textual change between {from_label} and {to_label}\n"

        file_path = os.path.join(self.output_dir, f"diff_iteration_{iteration}.diff")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(diff_text)

        return diff_text

    def summarise_diff(self, diff_text: str) -> str:
        """
        Return a one-line human-readable summary of a unified diff.
        Useful for REPORT.md notable-mutations section.
        """
        additions   = sum(1 for l in diff_text.splitlines() if l.startswith("+") and not l.startswith("+++"))
        deletions   = sum(1 for l in diff_text.splitlines() if l.startswith("-") and not l.startswith("---"))
        if additions == 0 and deletions == 0:
            return "No change."
        return f"+{additions} lines  -{deletions} lines"