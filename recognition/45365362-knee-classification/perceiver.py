from process_data import process_data

num_classes = 2
input_shape = (228, 260, 3)

X_train, y_train, X_test, y_test = process_data("AKOA_Analysis\AKOA_Analysis", 80, 20)

