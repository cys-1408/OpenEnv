import os

OUTPUT_FILE = "sample_output.txt"

# Folders to ignore
EXCLUDE_DIRS = {".git", "__pycache__", "venv", "node_modules", ".idea", ".vscode"}

# File types to ignore (optional)
EXCLUDE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".exe", ".dll", ".pyc"}


def should_skip(filepath):
    _, ext = os.path.splitext(filepath)
    return ext.lower() in EXCLUDE_EXTENSIONS


def write_project_structure(root_dir):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for subdir, dirs, files in os.walk(root_dir):
            # Remove excluded dirs
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                if file == OUTPUT_FILE:
                    continue

                filepath = os.path.join(subdir, file)

                if should_skip(filepath):
                    continue

                outfile.write(f"# ===== {filepath} =====\n")

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        outfile.write(f.read())
                except Exception:
                    outfile.write("[Binary or unreadable file]")

                outfile.write("\n\n")


if __name__ == "__main__":
    write_project_structure(".")
