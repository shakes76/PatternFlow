ind = 0

im = Image.open(x_sort_test[ind])

im1 = im
im_n = np.array(im1)
im_n_flat = im_n.reshape(-1, 1)

for j in range(im_n_flat.shape[0]):
    if im_n_flat[j] != 0:
        im_n_flat[j] = 255

s = data_transform(im)
pred = model_test(s.unsqueeze(0).cuda()).cpu()
pred = F.sigmoid(pred)
pred = pred.detach().numpy()

pred = threshold_predictions_p(pred)

it = Image.open(x_sort_testL[ind])

###########################################
# Plot images
###########################################
source = np.array(im)
print(source.shape)
plt.figure(figsize=(20, 5))
plt.subplot(1,5,1) 
plt.imshow(source)
plt.subplot(1,5,2) 
plt.imshow(data_transform3(im))
plt.subplot(1,5,3) 
plt.imshow(pred[0][0])
plt.subplot(1,5,4) 
plt.imshow(data_transform4(it))
plt.subplot(1,5,5) 
plt.imshow(it)


###########################################
# compute iou
###########################################
A = np.array(data_transform4(it))//255
B = pred[0][0]
intersection = (A * B).sum()
union = A.sum() + B.sum() - intersection
IOU = intersection / union
IOU


###########################################
# compute Dice coefficient
###########################################
dice = 2/(1/IOU + 1)
dice
