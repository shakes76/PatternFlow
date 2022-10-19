from tensorflow import keras
import pandas as pd


def handle_prediction(data, module):
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
