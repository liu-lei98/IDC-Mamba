def set_template(args):
    if args.tune:
        args.max_epoch = 20
        args.milestones = [7, 12, 15, 18]
        args.scheduler = 'MultiStepLR'

    if args.template.find('guan_guan') >= 0: 
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
        if int(args.method.split('_')[-1]) >= 9: args.batch_size = 1
        args.learning_rate = min(2e-4 * args.batch_size, 4e-4)

    if args.template.find('ISTMamba') >= 0: 
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
        if int(args.method.split('_')[-1]) >= 7: args.batch_size = 1
        args.learning_rate = min(4e-4 * args.batch_size, 4e-4)

    if args.template.find('HQSManba') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
        if int(args.method.split('_')[-1]) >= 7: args.batch_size = 1
        args.learning_rate = min(3e-4 * args.batch_size, 4e-4)

        
