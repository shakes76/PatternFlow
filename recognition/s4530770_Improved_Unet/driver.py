from data_loader import DataLoader
from improved_unet import build_model
from trainer import UnetTrainer

model = build_model((384, 512, 3), 64)
data = DataLoader("H:\\COMP3710\\ISIC2018_Task1-2_Training_Data", batch_size=1)

trainer = UnetTrainer(model, data)
trainer.train(30)
trainer.show_predictions(3)
trainer.plot_history()

