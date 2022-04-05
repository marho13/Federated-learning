from Mnist import mnistReader as mnist

x_train, y_train, x_test, y_test = mnist.getMnist()
mnist.show_images(x_train, y_train)
