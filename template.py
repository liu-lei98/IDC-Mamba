def set_template(args):
    if args.tune:
        args.max_epoch = 20
        args.milestones = [7, 12, 15, 18]
        args.scheduler = 'MultiStepLR'

    if args.template.find('IDCMamba') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
        if int(args.method.split('_')[-1]) >= 7: args.batch_size = 1
        args.learning_rate = min(3e-4 * args.batch_size, 4e-4)

        
