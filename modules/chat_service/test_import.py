import sys
from pathlib import Path

print("Current directory:", Path.cwd())
print("Script location:", Path(__file__).parent)

parent_dir = Path(__file__).parent.parent
print("Parent directory:", parent_dir)
sys.path.insert(0, str(parent_dir))

print("sys.path after insert:", sys.path[:3])
print("Directories in parent:", [d.name for d in parent_dir.iterdir() if d.is_dir()])

try:
    import assistant_service
    print("Successfully imported assistant_service")
    print("assistant_service location:", assistant_service.__file__)
except ImportError as e:
    print(f"Failed to import assistant_service: {e}")

try:
    from assistant_service import AssistantService
    print("Successfully imported AssistantService")
except ImportError as e:
    print(f"Failed to import AssistantService: {e}")