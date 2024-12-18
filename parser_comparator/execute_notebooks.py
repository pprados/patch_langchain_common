import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Union

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Packages nécessaires
REQUIRED_PACKAGES = ["nbformat", "nbconvert", "ipykernel"]


def ensure_global_packages(packages: list[str]) -> None:
    """S'assure que les packages nécessaires sont installés dans l'environnement global.
    Si ce n'est pas le cas, ils sont installés automatiquement.
    """
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            logging.info("Package '%s' non trouvé, installation en cours...", pkg)
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)


# S'assurer que l'environnement global dispose des packages requis
ensure_global_packages(REQUIRED_PACKAGES)

import nbformat
from nbconvert.preprocessors import (
    CellExecutionError,
    ExecutePreprocessor,
)

# Répertoire contenant les notebooks
NOTEBOOKS_DIR = Path("docs/docs")

# Nom de l'environnement virtuel utilisé pour l'exécution des notebooks
VENV_NAME = ".venv_notebooks"


def run_command(command: list[str]) -> None:
    """Exécute une commande système en vérifiant le code de retour."""
    logging.debug("Exécution de la commande : %s", " ".join(command))
    subprocess.run(command, check=True)


def remove_virtualenv(venv_path: Union[str, Path] = VENV_NAME) -> None:
    """Supprime l'environnement virtuel s'il existe."""
    venv_path = Path(venv_path)
    if venv_path.exists():
        logging.info("Suppression de l'environnement virtuel : %s", venv_path)
        shutil.rmtree(venv_path)


def create_virtualenv(venv_path: Union[str, Path] = VENV_NAME) -> None:
    venv_path = Path(venv_path)
    logging.info("Creat new virtual environment: %s", venv_path)
    run_command([sys.executable, "-m", "venv", str(venv_path)])
    run_command([str(venv_path / "bin" / "pip"), "install", "-q", "--upgrade", "pip"])
    run_command(
        [
            str(venv_path / "bin" / "pip"),
            "install",
            "-q",
            "nbformat",
            "nbconvert",
            "ipykernel",
        ]
    )


def execute_notebook(notebook_path: Union[str, Path]) -> None:
    """Exécute un notebook Jupyter en utilisant nbconvert et vérifie
    l'absence d'erreurs."""
    notebook_path = Path(notebook_path)
    logging.info("Exécution du notebook : %s", notebook_path)
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
        logging.info("Notebook %s exécuté avec succès.", notebook_path)
    except CellExecutionError as e:
        raise RuntimeError(
            f"Erreur lors de l'exécution du notebook {notebook_path}: {e}"
        )


def main() -> None:
    # Parcours récursif des notebooks
    notebook_paths = list(NOTEBOOKS_DIR.glob("**/*.ipynb"))
    if not notebook_paths:
        logging.warning("Aucun notebook trouvé dans le répertoire %s", NOTEBOOKS_DIR)
        return

    for notebook_path in notebook_paths:
        remove_virtualenv()
        create_virtualenv()
        try:
            execute_notebook(notebook_path)
        except RuntimeError as e:
            logging.error(str(e))
            sys.exit(1)  # Sortie du script avec code d'erreur


if __name__ == "__main__":
    main()
