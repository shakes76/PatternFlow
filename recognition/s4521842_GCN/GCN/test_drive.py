import matplotlib.pyplot as plt

from data_preprocessing import *
from model import *
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.manifold import TSNE

# crossentropy loss function for two or more label classes
loss_function = CategoricalCrossentropy(from_logits=True)


def calculate_loss(model, inputs, y_true, train_mask, training_state):
    y = model(inputs, training=training_state)
    # generate slices from y into a tensor
    y_pred = tf.gather_nd(y, tf.where(train_mask))
    # calculate loss based on categorical crossentropy loss
    return loss_function(y_true=y_true, y_pred=y_pred)


def train_step(model, inputs, labels, train_mask):
    with tf.GradientTape() as tape:
        loss = calculate_loss(model, inputs, labels, train_mask, training_state=True)

    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def predict(model, graph, mask):
    y = model(graph)
    y_pred = tf.gather_nd(y, tf.where(mask))
    return y_pred


def evaluate(model, mask, y_true, graph):
    y = model(graph)
    y_pred = tf.gather_nd(y, tf.where(mask))

    # Returns the truth if y_pred=y_true.
    ll = tf.math.equal(tf.math.argmax(y_true, -1), tf.math.argmax(y_pred, -1))
    # calculate accuracy
    accuarcy = tf.reduce_mean(tf.cast(ll, dtype=tf.float32))

    return accuarcy


def train(epochs, model, graph, train_mask, train_labels, val_labels):
    train_loss_ls = []
    train_acc_ls = []
    val_acc_ls = []

    for epoch in range(epochs):
        loss = train_step(model, graph, train_labels, train_mask)

        train_acc = evaluate(model, train_mask, train_labels, graph)
        val_acc = evaluate(model, val_mask, val_labels, graph)

        train_loss_ls.append(loss)
        train_acc_ls.append(train_acc)
        val_acc_ls.append(val_acc)

        print("Epoch_{}: loss={:.5f} tr_acc={:.5f} val_acc={:.5f}".format(
            epoch + 1, loss, train_acc, val_acc))

    return train_loss_ls, train_acc_ls, val_acc_ls


def plot_result(train_loss_ls, train_acc_ls, val_acc_ls):
    fig, axes = plt.subplots(3, figsize=(12, 8))
    fig.suptitle('GCN')

    axes[0].set_ylabel("Training Loss", fontsize=15)
    axes[0].plot(train_loss_ls)

    axes[1].set_ylabel("Training Accuracy", fontsize=15)
    axes[1].plot(train_acc_ls)

    axes[2].set_ylabel("Validation Accuracy", fontsize=15)
    axes[2].plot(val_acc_ls)

    plt.show()


def plot_TSNE(y_pred, labels, n_components=2, perplexity=30, init='pca', n_iter=3000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, init=init, n_iter=n_iter)
    data_low_dim = tsne.fit_transform(y_pred)
    plt.title('TSNE Visualisation')
    plt.scatter(data_low_dim[:, 0], data_low_dim[:, 1], marker='o', c=labels)
    plt.show()


# load data
dataset = DataPreprocessing()
(features, labels, adjacency,
 train_mask, val_mask, test_mask,
 train_labels, val_labels, test_labels) = dataset.get_data()

EPOCHS = 100
graph = [features, adjacency]
GCN_model = GCN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=5e-5)

if __name__ == '__main__':
    (train_loss_ls, train_acc_ls, val_acc_ls) = train(EPOCHS, GCN_model, graph, train_mask,
                                                      train_labels, val_labels)

    # evaluate accuracy by test dataset
    test_acc = evaluate(GCN_model, test_mask, test_labels, graph)
    print('Test Accuracy:', test_acc)

    plot_result(train_loss_ls, train_acc_ls, val_acc_ls)

    # plot t-sne
    y_pred = predict(GCN_model, graph, test_mask)
    labels_test = dataset.get_test_labels()
    plot_TSNE(y_pred, labels_test)



