import sys
import subprocess
from pathlib import Path


def convert_notebooks_to_py(root="examples"):
    """
    Convert Jupyter notebooks to Python scripts.
    """

    try:
        subprocess.run(
            ["jupyter", "nbconvert", "--version"], check=True, stdout=subprocess.PIPE
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "Error: jupyter nbconvert is not installed. Install with: pip install nbconvert"
        )
        return False

    root = Path(root)
    if not root.exists():
        print(f"Error: Directory '{root}' does not exist.")
        return False
    notebooks = [
        p for p in root.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts
    ]
    if not notebooks:
        print(f"No notebook files found under '{root}'.")
        return True
    print(f"Found {len(notebooks)} notebook(s) to convert:")
    ok = 0
    for nb in notebooks:
        try:
            rel = nb.relative_to(root)
        except ValueError:
            rel = nb.name
        out_dir = nb.parent
        py_path = out_dir / (nb.stem + ".py")
        print(f"  Converting: {rel} -> {py_path.name}")
        if py_path.exists():
            print(f"    (replacing existing {py_path.name})")
        try:
            cmd = [
                "jupyter",
                "nbconvert",
                "--to",
                "python",
                "--output",
                nb.stem,
                "--output-dir",
                str(out_dir),
                str(nb),
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Post-processing to remove get_ipython() lines
            if py_path.exists():
                with open(py_path, "r") as f:
                    lines = f.readlines()

                # Filter out lines containing the specific string
                modified_lines = [line for line in lines if "get_ipython()" not in line]

                with open(py_path, "w") as f:
                    f.writelines(modified_lines)

            print(f"    ✓ Successfully converted and cleaned {rel}")
            ok += 1
        except FileNotFoundError:
            print(
                "    ✗ Error: jupyter nbconvert not found. Install with: pip install nbconvert"
            )
            return False
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Error converting {rel}:\n      {e.stderr}")
    print(
        f"\nConversion complete: {ok}/{len(notebooks)} notebooks converted successfully."
    )
    return ok == len(notebooks)


def main():
    root = Path("/Users/elliottmacneil/python/chirpy/examples")
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    print(f"Converting notebooks under '{root}'...")
    success = convert_notebooks_to_py(root)
    if success:
        print("All conversions completed successfully!")
    else:
        print("Some conversions failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
