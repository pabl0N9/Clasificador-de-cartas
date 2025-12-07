from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

FOLDERS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "models",
    ROOT / "notebooks",
    ROOT / "tests",
    ROOT / "src" / "data",
    ROOT / "src" / "models",
    ROOT / "src" / "app",
    ROOT / "src" / "utils",
]

INIT_FOLDERS = [
    ROOT / "src",
    ROOT / "src" / "data",
    ROOT / "src" / "models",
    ROOT / "src" / "app",
    ROOT / "src" / "utils",
]

MOVES = {
    "data_preparation.py": ROOT / "src" / "data" / "prepare_data.py",
    "modelo_emociones.py": ROOT / "src" / "models" / "train.py",
    "app_interfaz.py": ROOT / "src" / "app" / "gui.py",
    "cartas_dataset.csv": ROOT / "data" / "processed" / "cartas_dataset.csv",
    "modelo_emociones.pkl": ROOT / "models" / "modelo_emociones.pkl",
    "label_encoder.pkl": ROOT / "models" / "label_encoder.pkl",
}

def make_dirs():
    for folder in FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)


def ensure_inits():
    for folder in INIT_FOLDERS:
        init_file = folder / "__init__.py"
        init_file.touch(exist_ok=True)


def safe_move(src_path: Path, dst_path: Path):
    if not src_path.exists():
        return
    if dst_path.exists():
        print(f"Skip {src_path.name}: destination already exists")
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_path), str(dst_path))
    print(f"Moved {src_path} -> {dst_path}")


def main():
    make_dirs()
    ensure_inits()
    for name, dst in MOVES.items():
        safe_move(ROOT / name, dst)


if __name__ == "__main__":
    main()
