from nns import KAN
from utils import evaluate_model, define_dimensions

input_size, hidden_size, num_classes = define_dimensions().values()

evaluate_model(KAN([input_size, hidden_size, num_classes]))
