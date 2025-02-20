from .IDCMamba import IDCMamba
def model_generator(opt, device="cuda"):
    method = opt.method 
   

    if 'IDCMamba' in method:
        num_iterations = int(method.split('_')[-1])
        # shallow_dim= 32
        if opt.debug:
            num_iterations = 2
        print(num_iterations)
        model = IDCMamba(num_iterations).to(device)
    else:
        print(f'opt.Method {opt.method} is not defined !!!!')
    
    return model