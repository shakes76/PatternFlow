from model.yolo_loss import YOLOLoss
from yolo_utiles.plot import *
from model.model import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from yolo_utiles.dataloader import YoloDataset, yolo_dataset_collate
from yolo_utiles.early_stop import EarlyStopping
import os
import driver

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Hyperparameter
epochs = 20
bs = 8
learning_rate = 0.001
num_workers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
last_var_loss = [0, ]

weights_path, anchors_mask, input_shape, class_names, num_classes, anchors, num_anchors, confidence, nms_iou, letterbox_image = driver.get_variable()


def fit(net, yolo_loss, opt, batch_data, batch_data_test):
	print("Train starts")
	trained_samples = 0
	all_samples = batch_data.dataset.__len__() * epochs
	iter_ = 0
	for e in range(38, 38 + epochs + 1):
		loss = 0
		test_loss = 0
		# train start
		net.train()
		for iteration, (images, y) in enumerate(batch_data):

			images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
			y = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in y]
			opt.zero_grad(set_to_none=True)
			outputs = net(images)

			loss_value_all = 0
			num_pos_all = 0
			# loop will run three times for the three different size of anchor box
			for current_anchor in range(len(outputs)):
				loss_item, num_pos = yolo_loss(current_anchor, outputs[current_anchor], y)
				loss_value_all += loss_item
				num_pos_all += num_pos
			loss_value = loss_value_all / num_pos_all

			trained_samples += images.shape[0]
			loss_value.backward()
			loss += loss_value.item()
			opt.step()

			if (iteration + 1) % 50 == 0:
				print('Epoch{}:[{}/{}({:.0f}%)]'.format(e, trained_samples, all_samples,
														100 * trained_samples / all_samples))

		net.eval()
		print("Test starts")
		for iteration, (images, y) in enumerate(batch_data_test):

			# prevent computational graph tracking
			with torch.no_grad():
				images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
				y = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in y]

				opt.zero_grad(set_to_none=True)
				outputs = net(images)
				loss_value_all = 0
				num_pos_all = 0
				# loop will run three times for the three different size of anchor box
				for current_anchor in range(len(outputs)):
					loss_item, num_pos = yolo_loss(current_anchor, outputs[current_anchor], y)
					loss_value_all += loss_item
					num_pos_all += num_pos
				loss_value = loss_value_all / num_pos_all
				test_loss += loss_value.item()

			# Update the length of the progress each time

			iter_ += 1
		lr_scheduler.step()
		epoch_train_loss = round(loss / (num_train // bs), 5)
		epoch_test_loss = round(test_loss / (num_test // bs), 5)

		early_stop = EarlyStopping()
		early_stop = early_stop(test_loss)
		if early_stop:
			print("Early stops")
			break
		if last_var_loss[0] > test_loss or e == 1:
			if e < 10:
				torch.save(model.state_dict(),
						   driver.weight_folder_path + '0{}epoch, training loss{},test_loss{}.pth'.format(
							   e, epoch_train_loss, epoch_test_loss))
				print("Weights Saved")
			else:
				torch.save(model.state_dict(),
						   driver.weight_folder_path + '{}epoch, training loss{},test_loss{}.pth'.format(
							   e, epoch_train_loss, epoch_test_loss))
				print("Weights Saved")
		last_var_loss[0] = test_loss
		print()
		print('Train loss{}, Test loss{}'.format(epoch_train_loss, epoch_test_loss))
		loss_list.append(epoch_train_loss)
		test_loss_list.append(epoch_test_loss)
		Plot_loss(loss_list, test_loss_list, [], epochs).plot_loss()


def init_train(net, bs, lr, input_shape, train_lines, num_classes, val_lines, num_workers):
	torch.manual_seed(1)
	opt = optim.SGD(net.parameters(), lr, weight_decay=1e-4)
	lr_scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
	train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=True)
	test_dataset = YoloDataset(val_lines, input_shape, num_classes, train=False)
	train = DataLoader(train_dataset, batch_size=bs, shuffle=True,
					   num_workers=num_workers, collate_fn=yolo_dataset_collate, drop_last=False, pin_memory=True)
	test = DataLoader(test_dataset, batch_size=bs, shuffle=False,
					  num_workers=num_workers, collate_fn=yolo_dataset_collate, drop_last=False, pin_memory=True)
	yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask)

	for param in net.darknet53.parameters():
		param.requires_grad = True
	return opt, lr_scheduler, yolo_loss, train_dataset, test_dataset, train, test


if __name__ == "__main__":
	loss_list = []
	test_loss_list = []

	if device:
		model = YoloBody(anchors_mask, num_classes).cuda()
	else:
		print("Do not have GPU")

	if weights_path != '':
		model.load_state_dict(torch.load(weights_path))

	num_train = len(driver.train_lines)
	num_test = len(driver.test_lines)

	if epochs > 0:
		opt, lr_scheduler, yolo_loss, train_dataset, test_dataset, train, test = init_train(model, bs,
																							learning_rate,
																							input_shape,
																							driver.train_lines,
																							num_classes,
																							driver.test_lines,
																							num_workers)

		fit(model, yolo_loss, opt, train, test)
