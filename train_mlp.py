from nns import MLP
from utils import evaluate_model, define_dimensions

if __name__ == "__main__":
    input_size, hidden_size, num_classes = define_dimensions().values()
    evaluate_model(MLP(input_size, hidden_size, num_classes))
