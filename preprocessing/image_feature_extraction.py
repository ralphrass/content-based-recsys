# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time
import pandas as pd

# display plots in this notebook
# matplotlib inline
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)  # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path; we'll add it here explicitly.
caffe_root = '/opt/caffe/'
model_root = '/home/ralph/models/ResNet/'
sys.path.insert(0, caffe_root + 'python')

import caffe

# if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#    print 'CaffeNet found.'

if os.path.isfile(model_root + 'ResNet-152-model.caffemodel'):
    print 'CaffeNet found.'

caffe.set_mode_gpu()

model_def = model_root + 'ResNet-152-deploy.prototxt'
model_weights = model_root + 'ResNet-152-model.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,  # batch size
                          3,  # 3-channel (BGR) images
                          224, 224)  # image size is 227x227 (AlexNet), 224x224 (ResNet)

# os.path.join("PASTA", "NOME_ARQUIVO")
# DIR = "/home/rfilho/temp/img/"
DIR = "/home/ralph/yelp_full_photo_dataset"
listaimg = os.listdir(DIR)

labels_file = '/home/ralph/caffe_synset/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')

COUNT = 0

image_dict = {}

start = time.time()

for imagem in listaimg:
    try:
        image = caffe.io.load_image(
            os.path.join(DIR, imagem))  # converte a imagem para ponto flutuante (scikit -> skimage -> img_as_float)
    except IOError:
        continue
    transformed_image = transformer.preprocess('data', image)

    # plt.show(transformed_image)

    net.blobs['data'].data[...] = transformed_image

    output = net.forward()

    output_prob = output['prob'][0]
    prob = output_prob[output_prob.argmax()]
    classe = labels[output_prob.argmax()]

    top_inds = output_prob.argsort()[::-1][:5]
    classes = [labels[c] for c in top_inds]
    probs = [output_prob[p] for p in top_inds]
    # print classes
    # print probs
    # exit()

    # probabilidade da maior classe, maior classe, 5 maiores classes, vetor de atributos
    image_dict[imagem.replace('.jpg', '')] = (prob, classe, classes, probs, net.blobs['pool5'].data.ravel())

    # print image_dict
    # break

    COUNT += 1

    if COUNT % 2000 == 0:
        print COUNT, "images read"

df = pd.DataFrame(image_dict)
df.to_pickle('yelp_200k_com_5_classes.pkl')

end = time.time()

print start - end, "seconds elapsed"

# image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
# image = caffe.io.load_image('~/Documentos/Dataset/Yelp/2016_yelp_dataset_challenge_photos/cat.jpg')
# image = caffe.io.load_image('/home/rfilho/temp/img/4-1.jpg') #converte a imagem para ponto flutuante (scikit -> skimage -> img_as_float)
# transformed_image = transformer.preprocess('data', image)
# plt.imshow(image)

# copy the image data into the memory allocated for the net
# net.blobs['data'].data[...] = transformed_image

# output = net.forward() ### perform classification
# output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
# print 'probabilities:', output_prob

# print 'predicted class is:', output_prob.argmax()

# load ImageNet labels
# labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
# labels = np.loadtxt(labels_file, str, delimiter='\t')
# print 'output label:', labels[output_prob.argmax()]

# strFeatures = ''.join(['%.8f,' % num for num in net.blobs['pool5'].data.flatten()])
# strFeatures = strFeatures[:-1] #Get rid of the last comma
# print VIstring

# with open('features.txt', 'w') as f:
#     np.savetxt(f, net.blobs['pool5'].data.copy(), fmt='%.8f', delimiter='\n')
# print 'Features:', net.blobs['res5c'].data.copy()




# sort top five predictions from softmax output
# top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
# print 'probabilities and labels:'
# zip(output_prob[top_inds], labels[top_inds])


# for layer_name, blob in net.blobs.iteritems():
#    print layer_name + '\t' + str(blob.data.shape)


'''import time

start = time.time()
net.forward()
end = time.time()
print(end-start)

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
net.forward()  # run once before timing to set up memory

start = time.time()
net.forward()
end = time.time()
print(end-start)


import os

my_image_url = "https://s3-media2.fl.yelpcdn.com/bphoto/oG7GVz03NJTOF9rpE4gFxQ/o.jpg"

os.system('wget -O image.jpg https://s3-media2.fl.yelpcdn.com/bphoto/oG7GVz03NJTOF9rpE4gFxQ/o.jpg')

# transform it and copy it into the net
image = caffe.io.load_image('image.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', image)

# perform classification
net.forward()

# obtain the output probabilities
output_prob = net.blobs['prob'].data[0]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]

plt.imshow(image)

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])'''