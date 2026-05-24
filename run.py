import sys
import io

# Force UTF-8 output on Windows (default console is cp1252 which crashes on
# Unicode arrows, emoji, and other characters used in progress output).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

from src.optimizer.loop import OptimizerLoop

def main():
    try:
        optimizer = OptimizerLoop(config_path="config/base_config.yaml")
        optimizer.run()
    except KeyboardInterrupt:
        print("\nRun interrupted by user. State is safely persisted in SQLite.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()