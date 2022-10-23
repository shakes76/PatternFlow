    adj, features, labels = load_data('facebook.npz')#returns normalized adjacency matrix, tensor features and labels
    features.shape[0]
    num_nodes=features.shape[0]
    #split data in semi supervised quatity, i.e train:set:test=20:20:60(since n_train<n_test)
    train_set = torch.LongTensor(range(int(num_nodes*0.2)))
    val_set = torch.LongTensor(range(int(num_nodes*0.2),int(num_nodes*0.4)))
    test_set = torch.LongTensor(range(int(num_nodes*0.4),num_nodes))

    model = GCN(in_feature=features.shape[1],
                out_class=len(np.unique(labels)), dropout=0.5)

    optimizer = optim.Adam([model.gcn_conv_1.weight,model.gcn_conv_2.weight], lr=0.01)
    train_model(200)
    test_model()
    train_accuracies = np.load('train_accuracies.npy')
    train_losses = np.load('train_losses.npy')
    validation_accuracies = np.load('validation_accuracies.npy')
    validation_losses = np.load('validation_losses.npy')
    plt.plot(range(200),train_accuracies,'b')
    plt.plot(range(200),validation_accuracies,'r')
    plt.legend(['train_accuracy', 'validation_accuracy'])
    plt.show()
    plt.plot(range(200),train_losses,'b')
    plt.plot(range(200),validation_losses,'r')
    plt.legend(['train_loss', 'validation_loss'])
    plt.show()
    #for plotting tsne map
    outputs = model.gcn_conv_1(features, adj).detach().numpy()
    plot_tsne(labels, outputs )
