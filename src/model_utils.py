import torch
from torch.autograd.functional import hessian, jacobian
from torch.func import functional_call
from src.models import *
from tqdm import tqdm

feature_dim = 15
sample_num = 20

model = MLP(feature_dim, 1)
inp = torch.rand(sample_num, feature_dim)

# return torch.one array jacob
def jacob_format(result_jacob):
    transform_jacobs = []
    for i in range(len(result_jacob)):
        transform_jacob = result_jacob[i].reshape(1, -1)
        if len(transform_jacob.size()) > 1:
            transform_jacobs.append(transform_jacob[0])
        else:
            transform_jacobs.append(transform_jacob)
    node_nums = [len(transform_jacobs[i]) for i in range(len(result_jacob))]
    output_jacob = torch.concat([transform_jacobs[i] for i in range(len(result_jacob))])
    return node_nums, output_jacob

def hessian_format(node_nums, result_hessian):
    a = []
    for j, node_num in enumerate(node_nums):
        a.append(torch.concat([result_hessian[j][i].reshape(node_num, -1) for i, __ in enumerate(node_nums)], dim = 1))
    final_hessian = torch.concat([a[i] for i in range(len(node_nums))])
    return final_hessian

def nn_gradient_hessian(model, inp, target_out):
    model = model.to(set_device)
    inp = inp.to(set_device)
    target_out = target_out.to(set_device)
    criterion = torch.nn.MSELoss()
    print(set_device)

    print('compute gradients and hessians of the inputs')
    print('==================================')
    gradients = []
    empirical_hessian = 0

    sample_num = len(inp)
    for i in tqdm(range(sample_num)):
        def loss_i(*params):
            out: torch.Tensor = functional_call(model, {n: p for n, p in zip(names, params)}, inp[i])
            
            return torch.square(out.to(set_device) - torch.Tensor([target_out[i]]).float().to(set_device))

        names = list(n for n, _ in model.named_parameters())
        
        result_jacob = jacobian(loss_i, tuple(model.parameters()))
        node_nums, a = jacob_format(result_jacob)
        a = a.to(set_device)
        result_hessian = hessian(loss_i, tuple(model.parameters()))
        b = hessian_format(node_nums, result_hessian)
        gradients.append(a)
        empirical_hessian += b / sample_num
    
   
    empirical_hessian = empirical_hessian.to(set_device)
    return gradients, empirical_hessian
    

