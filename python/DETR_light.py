import torch


class DETR(torch.nn.Module):

    def __init__(self, num_classes=91, hidden_dim=100, nheads=10,
                 num_encoder_layers=2, num_decoder_layers=2, backboneChannels=256):
        super().__init__()
        self.backbone = Convolution(backboneChannels)
        self.conv = torch.nn.Conv2d(backboneChannels, hidden_dim, 1)

        self.transformer = torch.nn.Transformer(hidden_dim, nheads,
                                                num_encoder_layers, num_decoder_layers)

        self.linear_class = torch.nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = torch.nn.Linear(hidden_dim, 4)

        self.query_pos = torch.nn.Parameter(torch.rand(100, hidden_dim))

        self.row_embed = torch.nn.Parameter(torch.rand(100, hidden_dim // 2))
        self.col_embed = torch.nn.Parameter(torch.rand(100, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)

        H, W = h.shape[-2:]

        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1))

        h = h.permute(1, 0, 2)

        classes = self.linear_class(h)
        classes = torch.nn.functional.softmax(classes, dim=2)

        boxes = self.linear_bbox(h)
        boxes = torch.sigmoid(boxes)

        return classes, boxes


class Convolution(torch.nn.Module):

    def __init__(self, finalChannels):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            3, finalChannels//4, 11, stride=5, dilation=1)
        self.conv2 = torch.nn.Conv2d(
            finalChannels//4, finalChannels//2, 9, stride=3, dilation=1)
        self.conv3 = torch.nn.Conv2d(
            finalChannels//2, finalChannels, 7, stride=1, dilation=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        return x
