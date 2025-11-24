import torch
from agentic_pm.modeling.models import GRURegressor

def test_gru_reproducibility():
    torch.manual_seed(123)
    model1 = GRURegressor(input_dim=5, hidden_dim=6, num_layers=1)
    torch.manual_seed(123)
    model2 = GRURegressor(input_dim=5, hidden_dim=6, num_layers=1)

    for (n1,p1), (n2,p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2)
