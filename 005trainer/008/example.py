from chainer import serializers

# Create an instance of the network you trained
model = MyNetwork()

# Load the saved paremeters into the instance
serializers.load_npz('my_mnist.model', model)

# Get a test image and label
x, t = test[0]
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.savefig('7.png')
print('label:', t)
