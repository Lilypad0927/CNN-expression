import numpy as np

labels = []  # labels are of the form [NEU HAP SAD SUR ANG DIS FEA] = neutral happy sad surprise angry disgust fear
text = open('data/jaffe_labels/JAFFE_labels.txt')
for line in text.readlines()[2:]:
    line_label = line[-4:-2]  # e.g. NE HA SA SU AN DI FE
    tag = line [-7:-1]  # e.g. YM-AN3
    index = 0
    if line_label == 'NE':
        index = 6
    if line_label == 'HA':
        index = 3
    if line_label == 'SA':
        index = 4
    if line_label == 'SU':
        index = 5
    if line_label == 'AN':
        index = 0
    if line_label == 'DI':
        index = 1
    if line_label == 'FE':
        index = 2
    labels.append([tag,index])  # labels e.g. [YM-AN3, 0]
print labels
label_tensor = np.array(labels)
print label_tensor
print len(label_tensor)
i = np.argsort(label_tensor[:,0])
print i
print len(i)
label_tensor = label_tensor[i]
print label_tensor
print len(label_tensor)
label_tensor = label_tensor[:,1]
print label_tensor
print len(label_tensor)
label_tensor = label_tensor.astype(int)
print label_tensor
print len(label_tensor)
one_hot = np.zeros([np.size(label_tensor), 7])  # 0 matrix [label_tensor*7]
print one_hot
for i in range(np.size(label_tensor)):
    one_hot[i, label_tensor[i]] = 1
print one_hot