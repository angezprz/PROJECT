from src.data_preprocessing import load_dataset
from src.windowing import harmonize_and_window
from src.model import build_lstm_model
from src.train import evaluate_model

def main():
    # load
    df = load_dataset("data/raw/")
    # window
    windows_meta, idx_lists = harmonize_and_window(df, window_s=3.0, step_s=1.5)
    # build model
    model = build_lstm_model((100, 4), num_classes=5)
    # TODO: training logic here
    print("Pipeline executed successfully.")

if __name__ == "__main__":
    main()
