import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimpleSimCLR(nn.Modlue):
    class ResNetSimCLR(nn.Module):

        def __init__(self, encode_size, out_dim):
            super(SimpleSimCLR, self).__init__()

            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.relu2 = nn.ReLU()
            self.max_pool = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(9216, encode_size)
            self.relu3 = nn.ReLU()

            # projection MLP

            self.l1 = nn.Linear(encode_size, encode_size)
            self.l2 = nn.Linear(encode_size, out_dim)


        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.max_pool(x)
            x = self.flatten(x)
            x = self.fc1(x)
            h = self.relu3(x)
            # Projection before loss
            x = self.l1(h)
            x = F.relu(x)
            x = self.l2(x)

            return h, x

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
