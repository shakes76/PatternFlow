from tensorflow import keras
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


def handle_prediction(data, module):
    """
    Takes the dataset class and module class to test the trained model
    on every facebook class. The predicted results is saved into a csv file
    and the TSNE plot is saved as a png file

    Args:
        data: the dataset class used to varify the prediction
        module: the module class that contains the model values

    """
    new_model = keras.models.load_model("finalised_model")
    all_nodes = data.get_node_features().index
    all_gen = module.get_generator().flow(all_nodes)
    try_predict = new_model.predict(all_gen)
    node_predictions = module.get_target_encoding().inverse_transform(
        try_predict.squeeze()
    )
    predicted_result = pd.DataFrame({
        "Predicted": node_predictions,
        "True": data.get_target()["page_type"]
    })
    predicted_result.to_csv("Prediction_vs_Real", sep='\t')
    print(predicted_result)
    matrix = try_predict.squeeze(0)
    tsne = TSNE(n_components=2)
    reduced_matrix = tsne.fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(7, 7))
    cat_features = data.get_target()["page_type"].astype("category")
    ax.scatter(
        reduced_matrix[:, 0],
        reduced_matrix[:, 1],
        c=cat_features.cat.codes,
        cmap="jet",
        s=5,
        alpha=0.5,
    )
    ax.set(
        aspect="equal",
        title="TSNE of model"
    )
    plt.savefig("TSNE.png")
    plt.show()
