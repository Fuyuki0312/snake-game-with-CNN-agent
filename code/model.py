from torch import nn

class CNN_QLearning(nn.Module):

    def __init__(self,
                 in_channels=4*3,
                 hidden_channels=16,
                 kernel_size=5,
                 stride=1,
                 padding=2,
                 max_pool_size=2,
                 adaptive_pool_size=(9, 9),
                 final_channels=3):

        super().__init__()


        self.conv_block1 = nn.Sequential(

            nn.Conv2d(in_channels=in_channels,out_channels=hidden_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=max_pool_size)
        )


        hidden_channels_2 = hidden_channels * 2

        self.conv_block2 = nn.Sequential(

            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels_2,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_channels_2, out_channels=hidden_channels_2,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=max_pool_size)
        )


        hidden_channels_3 = hidden_channels_2 * 2

        self.conv_block3 = nn.Sequential(

            nn.Conv2d(in_channels=hidden_channels_2, out_channels=hidden_channels_3,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_channels_3, out_channels=hidden_channels_3,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=max_pool_size)
        )


        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(adaptive_pool_size),
            nn.Flatten(),
            nn.Linear(in_features=hidden_channels_3 * adaptive_pool_size[0] * adaptive_pool_size[1],
                      out_features=final_channels)
        )


    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

