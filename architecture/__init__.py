from .ReModelG import ISTMamba
from .HQSManbaPure import HQSManbaOri
def model_generator(opt, device="cuda"):
    method = opt.method 
   
    if 'ISTMamba' in method:
        num_iterations = int(method.split('_')[-1])
        # shallow_dim= 32
        if opt.debug:
            num_iterations = 2
        print(num_iterations)
        model = ISTMamba(dim=28,stage=num_iterations, shared = opt.shared).to(device)
    elif 'HQSManbaOri' in method:
        num_iterations = int(method.split('_')[-1])
        # shallow_dim= 32
        if opt.debug:
            num_iterations = 2
        print(num_iterations)
        model = HQSManbaOri(num_iterations).to(device)
    # elif 'HQSManba' in method:
    #     num_iterations = int(method.split('_')[-1])
    #     # shallow_dim= 32
    #     if opt.debug:
    #         num_iterations = 2
    #     print(num_iterations)
    #     model = HQSManba(num_iterations).to(device)
        # print(shallow_dim)
    else:
        print(f'opt.Method {opt.method} is not defined !!!!')
    
    return model