from vit_tensorflow.vit import ViT

def get_model(image_size, patch_size=32, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emp_dropout=0.1):
    return ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = 2,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = dropout,
        emb_dropout = emp_dropout
    )

if __name__ == "__main__":
    from dataset import get_data_preprocessing

    cropped_image_size = (192, 160)
    train_ds, test_ds, preprocessing = get_data_preprocessing(image_size=(240, 256), cropped_image_size=cropped_image_size, cropped_pos=(20, 36))
    model = get_model(max(cropped_image_size))