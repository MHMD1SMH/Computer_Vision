from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

model = YOLO(BASE_DIR / "Training/football_players-2/weights/best.pt")

results = model.predict(
    BASE_DIR / "Input_Videos/08fd33_4.mp4",
    save=True,
    show=True,
    device=0,
    project=BASE_DIR / "runs",
    name="football_test"
)
