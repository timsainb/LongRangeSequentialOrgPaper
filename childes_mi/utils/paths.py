from pathlib2 import Path
import pathlib2
import os

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

CHILDES_DIR = DATA_DIR / "raw/NLTK_Data_Dir/corpora"
CHILDES_DFS = DATA_DIR / "processed/childes/"

PHONBANK_DIR = DATA_DIR / "raw/PHON_Data_Dir/corpora"
PHONBANK_DFS = DATA_DIR / "processed/phonbank/"

MPIICOOKING2_DIR = DATA_DIR / "raw/MPIICOOKING2_Data_Dir/"

BREAKFAST_DIR = DATA_DIR / "raw/BREAKFAST_Data_Dir/"

EPIC_KITCHENS_DIR = DATA_DIR / "raw/EPIC_KITCHENS_Data_Dir/"

ZEBRAFISH_DIR = DATA_DIR / "raw/ZEBRAFISH_Data_Dir/"

DROSOPHILA_DIR = DATA_DIR / "raw/DROSOPHILA_Data_Dir/"


FIGURE_DIR = PROJECT_DIR/ "figures"

def ensure_dir(file_path):
    """ create a safely nested folder
    """
    if type(file_path) == str:
        if "." in os.path.basename(os.path.normpath(file_path)):
            directory = os.path.dirname(file_path)
        else:
            directory = os.path.normpath(file_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except FileExistsError as e:
                # multiprocessing can cause directory creation problems
                print(e)
    elif type(file_path) == pathlib2.PosixPath:
        # if this is a file
        if len(file_path.suffix) > 0:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path.mkdir(parents=True, exist_ok=True)
