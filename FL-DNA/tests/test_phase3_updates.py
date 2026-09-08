"""Check local-update replay against native Adam and independent derivatives."""
import copy

import pytest
import torch

from attacks.local_update import simulate, simulate_sgd, objective
from experiments.fraud_fl_common import BinaryFocalLoss
from models.fraud_mlp import FraudMLP


@pytest.mark.parametrize('steps', [1, 2, 5])
def test_native_adam_replay_and_rng_isolation(steps):
    torch.set_num_threads(1)
    torch.manual_seed(7351)
    model = FraudMLP(7).train()
    original = copy.deepcopy(model.state_dict())
    x, y = torch.randn(8,7), torch.tensor([[0.],[1.]]*4)
    batches = [slice(0,8)]*steps
    criterion = BinaryFocalLoss()
    rng = torch.get_rng_state().clone()
    local = copy.deepcopy(model)
    opt = torch.optim.Adam(local.parameters(),lr=.001)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng)
        for ids in batches:
            opt.zero_grad()
            criterion(local(x[ids]),y[ids]).backward()
            opt.step()
    replay = simulate(model,criterion,x,y,batches,rng)
    for key,value in local.state_dict().items():
        torch.testing.assert_close(replay[key],value-original[key],atol=2e-7,rtol=2e-4)
        assert torch.equal(model.state_dict()[key],original[key])
    assert torch.equal(torch.get_rng_state(),rng)


def test_input_derivative_matches_finite_difference():
    torch.manual_seed(6318)
    model = torch.nn.Linear(3,1).double().train()
    x = torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    y = torch.randn(4,1,dtype=torch.float64)
    criterion = torch.nn.MSELoss()
    rng = torch.get_rng_state()
    batches = [slice(0,2),slice(2,4)]
    signal = {k:torch.zeros_like(v) for k,v in model.named_parameters()}
    def evaluate(value):
        return objective(simulate(model,criterion,value,y,batches,rng),signal,list(signal))
    analytic, = torch.autograd.grad(evaluate(x),x)
    direction = torch.randn_like(x)
    epsilon = 1e-5
    numeric = (evaluate(x.detach()+epsilon*direction)-evaluate(x.detach()-epsilon*direction))/(2*epsilon)
    torch.testing.assert_close((analytic*direction).sum(),numeric,atol=1e-10,rtol=1e-3)
    assert analytic.abs().max()>0


def test_native_sgd_replay_with_batchnorm_buffers():
    torch.set_num_threads(1)
    torch.manual_seed(9531)
    model = FraudMLP(7).train()
    original = copy.deepcopy(model.state_dict())
    x, y = torch.randn(4, 7), torch.tensor([[0.], [1.], [0.], [1.]])
    criterion = BinaryFocalLoss()
    rng = torch.get_rng_state().clone()
    native = copy.deepcopy(model)
    optimizer = torch.optim.SGD(native.parameters(), lr=.001)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng)
        optimizer.zero_grad(); criterion(native(x), y).backward(); optimizer.step()
    replay = simulate_sgd(model, criterion, x, y, [slice(0, 4)], rng)
    for key, value in native.state_dict().items():
        torch.testing.assert_close(replay[key], value-original[key], atol=2e-7, rtol=2e-4)


@pytest.mark.parametrize('replay', [simulate_sgd, simulate])
def test_sgd_batchnorm_buffer_derivative(replay):
    torch.manual_seed(983)
    model = FraudMLP(7).double().train()
    x = torch.randn(4, 7, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1.], [0.], [0.], [1.]], dtype=torch.float64)
    rng = torch.get_rng_state()
    def evaluate(value):
        delta = replay(model, BinaryFocalLoss(), value, y, [slice(0, 4)], rng)
        return sum(v.square().sum() for k, v in delta.items() if 'running_' in k)
    grad, = torch.autograd.grad(evaluate(x), x)
    direction = torch.randn_like(x)
    epsilon = 1e-6
    numeric = (evaluate(x.detach()+epsilon*direction)-evaluate(x.detach()-epsilon*direction))/(2*epsilon)
    torch.testing.assert_close((grad*direction).sum(), numeric, rtol=1e-4, atol=1e-7)
    assert grad.abs().max() > 0
    assert all(not module._forward_pre_hooks for module in model.modules())
