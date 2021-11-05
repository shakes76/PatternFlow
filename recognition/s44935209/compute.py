ious = []

dices = []

for i in range(len(x_sort_test)):
    im = Image.open(x_sort_test[i])

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
    
    it = Image.open(x_sort_testL[i])
    A = np.array(data_transform4(it))//255
    B = pred[0][0]
    
    intersection = (A * B).sum()
    union = A.sum() + B.sum() - intersection
    iou = intersection / union
    ious.append(iou)
    dice = 2/(1/iou + 1)
    dices.append(dice)
    
    
    
print("average IOU", sum(ious)/len(ious))
print("average DICE", sum(dices)/len(dices))
