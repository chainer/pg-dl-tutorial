import chainer
from chainer import datasets
from chainer import links as L
from chainer import functions as F
from chainer import Variable, optimizers
from chainer import training
from chainer.training import extensions
import numpy as np
import argparse
import math
import random

class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class Block(chainer.Chain):
    def __init__(self, in_ch, out_ch, down_sampling):
        w = math.sqrt(2)
        stride = 2 if down_sampling else 1
        super(Block, self).__init__(
            bn1=L.BatchNormalization(in_ch),
            c1=L.Convolution2D(in_ch, out_ch, 3, pad=1, stride=stride, wscale=w, nobias=True),
            bn2=L.BatchNormalization(out_ch),
            c2=L.Convolution2D(out_ch, out_ch, 3, pad=1, stride=1, wscale=w, nobias=True)
        )
        if in_ch != out_ch:
            self.add_link("proj", L.Convolution2D(in_ch, out_ch, 3, pad=1, stride=2, wscale=w, nobias=True))

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_proj = in_ch != out_ch

    def __call__(self, x, train):
        h = self.c1(F.elu(self.bn1(x, test=not train)))
        h = self.c2(F.elu(self.bn2(h, test=not train)))
        shortcut = self.proj(x) if self.use_proj else x
                          
        return h + shortcut
        

class Group(chainer.Chain):
    def __init__(self, n_layers, in_ch, out_ch, down_sampling=False):
        super(Group, self).__init__()
        for i in xrange(n_layers):
            ch = in_ch if i == 0 else out_ch
            self.add_link('b{}'.format(i+1), Block(ch, out_ch, down_sampling and i == 0))

    def __call__(self, x, train):
        for l in self.children():
            x = l(x, train)
        return x
        

class ResNet(chainer.Chain):

    def __init__(self, n_out, n_layers=9):
        w = math.sqrt(2)
        super(ResNet, self).__init__(
            conv1=L.Convolution2D(3, 16, 3, pad=1, stride=1, wscale=w, nobias=True),
            conv2=Group(n_layers, 16, 16, False),
            conv3=Group(n_layers, 16, 32, True),
            conv4=Group(n_layers, 32, 64, True),
            bn=L.BatchNormalization(64),
            fc=L.Linear(64, n_out)
        )
        self.train = True

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h, self.train)
        h = self.conv3(h, self.train)
        h = self.conv4(h, self.train)
        h = F.elu(self.bn(h))
        h = F.average_pooling_2d(h, 8, stride=1)
        h = self.fc(h)
        return h

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, base, crop_size=4, random=True):
        self.base = base
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(-4, 4)
            left = random.randint(-4, 4)
            bottom = top + h
            right = left + w
            new_image = np.zeros_like(image)
            if top < 0:
                if left < 0:
                    new_image[:, -top:h, -left:w] = image[:, 0:bottom, 0:right]
                else:
                    new_image[:, -top:h, 0:w-left] = image[:, 0:bottom, left:w]
            else:
                if left < 0:
                    new_image[:, 0:h-top, -left:w] = image[:, top:h, 0:right]
                else:
                    new_image[:, 0:h-top, 0:w-left] = image[:, top:h, left:w]
            image = new_image

            if random.randint(0, 1):
                image = image[:, :, ::-1]

        mean = image.mean(axis=(1, 2))
        var = image.var(axis=(1, 2))
        var += 1e-10
        image -= mean[:, np.newaxis, np.newaxis]
        image /= var[:, np.newaxis, np.newaxis]
        return image, label

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

train, test = chainer.datasets.get_cifar10()
pre_train = PreprocessedDataset(train)
pre_test = PreprocessedDataset(test, False)

batchsize = 128
train_iter =chainer.iterators.SerialIterator(pre_train, batchsize)
test_iter = chainer.iterators.SerialIterator(pre_test, batchsize,
                                             repeat=False, shuffle=False)

# model = L.Classifier(MLP(784, 10))
model = L.Classifier(ResNet(10, 4))

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU

opt = chainer.optimizers.MomentumSGD(lr=0.1, momentum=0.9)
opt.use_cleargrads()
opt.setup(model)
opt.add_hook(chainer.optimizer.WeightDecay(0.0005))

epoch = 200

# Set up a trainer
updater = training.StandardUpdater(train_iter, opt, device=args.gpu)
trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

# Take a snapshot at each epoch
trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy']))

# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

resume = False
if resume:
    # Resume from a snapshot
    chainer.serializers.load_npz(resume, trainer)

# Run the training
trainer.run()
