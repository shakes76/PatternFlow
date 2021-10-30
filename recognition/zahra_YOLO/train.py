def train_model(epochs,img_data,batch_size,model_conf_path,data_conf_path,weights_path,classnames_path,n_cpu,img_size,checkpoint_interval,checkpoint_dir,use_cuda):
    cuda = torch.cuda.is_available() and use_cuda
    os.makedirs("checkpoints", exist_ok=True)
    classes = load_classes(classnames_path)

# Get data configuration
    data_config = parse_data_config(data_conf_path)
    train_path = data_config["train"]

# Get model parameters
    params = parse_model_config(model_conf_path)[0]
    learning_rate = float(params["learning_rate"])
    momentum = float(params["momentum"])
    decay = float(params["decay"])
    burn_in = int(params["burn_in"])
# Initiate darknet model
    model = Darknet(model_conf_path)
    model.load_weights(weights_path)
    if cuda:
        model = model.cuda()

    model.train()

# run torch dataloader which initiae data preprocessing and loading data
    dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=batch_size, shuffle=False, num_workers=n_cpu
)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

#use a PyTorch Variable which is a wrapper around a PyTorch Tensor, and represents a node in a computational graph. 

    for epoch in range(epochs):
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)


#PyTorch accumulates the gradients on backward passes.so we need to explicitly set the gradients to zero before start training to make sure that gradients points in the intended direction
            optimizer.zero_grad()
            loss = model(imgs, targets)
            loss.backward()
            optimizer.step()
            model.seen += imgs.size(0)
        model.save_weights("%s/%d.weights" % (checkpoint_dir, epoch))
#after each epoch, model saves the weights. we should take the last weight and use for prediction        