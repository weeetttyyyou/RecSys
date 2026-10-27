from pathlib import Path
import nbformat
from nbclient import NotebookClient
from ipykernel.kernelspec import install

root = Path(__file__).resolve().parents[1]
install(user=True, kernel_name="recsys-stas", display_name="Python recsys-stas")
notebook = nbformat.read(root / "experiments.ipynb", as_version=4)
NotebookClient(notebook, timeout=None, kernel_name="recsys-stas", resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(notebook, root / "experiments.ipynb")
print("Ноутбук выполнен")
