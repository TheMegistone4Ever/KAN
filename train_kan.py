from nns import KAN
from utils import evaluate_model, define_dimensions

if __name__ == "__main__":
    input_size, hidden_size, num_classes = define_dimensions().values()
    evaluate_model(KAN([input_size, hidden_size, num_classes]))
