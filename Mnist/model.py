from torch import nn
from torch.nn import functional as F
import torch

class Resnet20(nn.Module):
    """Small ConvNet for MNIST."""

    def __init__(self, stateDim, outputSize, n_latent_var):
        super().__init__()
        # self.device = device
        self.randPolicy = {"Rand": 0, "Policy": 0}
        self.current_step = 0
        self.num_actions = outputSize

        self.conv1 = nn.Conv2d(in_channels=stateDim, out_channels=n_latent_var, kernel_size=(7,7), stride=(2,2))
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=n_latent_var, out_channels=n_latent_var, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(in_channels=n_latent_var, out_channels=n_latent_var, kernel_size=(3,3))
        self.conv4 = nn.Conv2d(in_channels=n_latent_var, out_channels=n_latent_var, kernel_size=(3,3))
        self.conv5 = nn.Conv2d(in_channels=n_latent_var, out_channels=n_latent_var, kernel_size=(3,3))
        self.conv6 = nn.Conv2d(in_channels=n_latent_var, out_channels=n_latent_var*2, kernel_size=(3,3))
        self.conv7 = nn.Conv2d(in_channels=n_latent_var*2, out_channels=n_latent_var*2, kernel_size=(3,3))
        self.conv8 = nn.Conv2d(in_channels=n_latent_var*2, out_channels=n_latent_var*2, kernel_size=(3,3))
        self.conv9 = nn.Conv2d(in_channels=n_latent_var*2, out_channels=n_latent_var*2, kernel_size=(3,3))
        self.conv10 = nn.Conv2d(in_channels=n_latent_var*2, out_channels=n_latent_var*4, kernel_size=(3,3))
        self.conv11 = nn.Conv2d(in_channels=n_latent_var*4, out_channels=n_latent_var*4, kernel_size=(3,3))
        self.conv12 = nn.Conv2d(in_channels=n_latent_var*4, out_channels=n_latent_var*4, kernel_size=(3,3))
        self.conv13 = nn.Conv2d(in_channels=n_latent_var*4, out_channels=n_latent_var*4, kernel_size=(3,3))
        self.conv14 = nn.Conv2d(in_channels=n_latent_var*4, out_channels=n_latent_var*8, kernel_size=(3,3))
        self.conv15 = nn.Conv2d(in_channels=n_latent_var*8, out_channels=n_latent_var*8, kernel_size=(3,3))
        self.conv16 = nn.Conv2d(in_channels=n_latent_var*8, out_channels=n_latent_var*8, kernel_size=(3,3))
        self.conv17 = nn.Conv2d(in_channels=n_latent_var*8, out_channels=n_latent_var*8, kernel_size=(3,3))
        self.pool2 = nn.AvgPool2d(kernel_size=(3,3), stride=(2,2))
        self.fc1 = nn.Linear(in_features=n_latent_var*8*10*10, out_features=1000)
        self.output = nn.Linear(in_features=1000, out_features=outputSize)


    def forward(self, x): #Resnet20 no residual block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.output(x)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

class tinyNet(nn.Module):
    """Small ConvNet for MNIST."""

    def __init__(self, stateDim, outputSize, n_latent_var):
        super().__init__()
        # self.device = device
        self.randPolicy = {"Rand": 0, "Policy": 0}
        self.current_step = 0
        self.num_actions = outputSize

        self.conv1_1 = nn.Conv2d(in_channels=stateDim, out_channels=32, kernel_size=(1,1))
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))

        self.conv2_1 = nn.Conv2d(in_channels=stateDim, out_channels=32, kernel_size=(1,1))
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5))

        self.pool3_1 = nn.MaxPool2d(kernel_size=(3,3))
        self.conv3_2 = nn.Conv2d(in_channels=stateDim, out_channels=64, kernel_size=(1,1))

        self.conv4_1 = nn.Conv2d(in_channels=stateDim, out_channels=64, kernel_size=(1,1))

        self.fc1 = nn.Linear(in_features=135488, out_features=1000)
        self.output = nn.Linear(in_features=1000, out_features=outputSize)


    def forward(self, x): #InceptionBlock no residual block
        block1 = self.conv1_1(x)
        block1 = torch.flatten(self.conv1_2(block1), start_dim=1)

        block2 = self.conv2_1(x)
        block2 = torch.flatten(self.conv2_2(block2), start_dim=1)

        block3 = self.pool3_1(x)
        block3 = torch.flatten(self.conv3_2(block3), start_dim=1)

        block4 = torch.flatten(self.conv4_1(x), start_dim=1)

        # print(block1.shape, block2.shape, block3.shape, block4.shape)
        cat = torch.cat((block1, block2, block3, block4), dim=1)
        # print(cat.shape)

        x = torch.flatten(cat, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.output(x)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)