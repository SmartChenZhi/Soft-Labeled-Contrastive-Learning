import segmentation_models_pytorch as smp
from torch.nn import functional as F
from torch import nn

from utils.utils_ import get_n_params


class segmentation_models(nn.Module):
    def __init__(self, name='resnet50', pretrained=False, decoder_channels=(512, 256, 128, 64, 32), in_channel=3,
                 classes=4, multilvl=False, args=None):
        super(segmentation_models, self).__init__()
        self.multilvl = multilvl
        model = smp.Unet(
            encoder_name=name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet' if pretrained else None,
            decoder_channels=decoder_channels,
            in_channels=in_channel,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=classes,  # model output channels (number of classes in your dataset)
        )
        self.encoder = model.encoder
        self.decoder = model.decoder
        if args is not None:
            if 'phead' in vars(args):
                self.project_head = args.phead
            else:
                self.project_head = False
        else:
            self.project_head = False
        if self.project_head:
            self.phead = nn.Sequential(*[nn.Conv2d(decoder_channels[-1], decoder_channels[-1] * 2, kernel_size=1), nn.ReLU(),
                                        nn.Conv2d(decoder_channels[-1] * 2, decoder_channels[-1], kernel_size=1)])
        self.classifier = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=classes, kernel_size=(1, 1))
        if self.multilvl:
            self.classifier_aux = nn.Conv2d(in_channels=decoder_channels[-2], out_channels=classes, kernel_size=(1, 1))
        print(f'Model {name} loaded.')
        print(f'Number of params: {get_n_params(self):,}')

    def forward(self, x, features_out=True):
        features = self.encoder(x)  # channels [3, 64, 256, 512, 1024, 2048]
        """decoder forward"""
        output = features[1:]  # remove first skip with same spatial resolution
        output = output[::-1]  # reverse channels to start from head of encoder
        head = output[0]
        skips = output[1:]
        decoder_output = self.decoder.center(head)
        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            target_height = skip.size(-2) if skip is not None else decoder_output.size(-2) * 2
            target_width = skip.size(-1) if skip is not None else decoder_output.size(-1) * 2
            decoder_output = decoder_block(decoder_output, target_height, target_width, skip_connection=skip)
            if self.multilvl and (i == len(self.decoder.blocks) - 2):
                output_aux = decoder_output
                output_aux = F.interpolate(output_aux, size=x.size()[2:], mode='bilinear', align_corners=True)
                output_aux = self.classifier_aux(output_aux)
        # decoder_output = self._decoder(*features)
        output = self.classifier(decoder_output)
        if self.project_head:
            decoder_output = self.phead(decoder_output)
        if self.multilvl:
            return output, output_aux, decoder_output
        elif features_out:
            return output, features[-1], decoder_output
        else:
            return output


class PointNet(nn.Module):
    def __init__(self, num_points=300, fc_inch=81, conv_inch=512, ext=False):
        super().__init__()
        self.num_points = num_points
        self.ReLU = nn.LeakyReLU(inplace=True)
        # Final convolution is initialized differently form the rest
        if ext:
            self.conv1 = nn.Conv2d(conv_inch, conv_inch * 2, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(conv_inch * 2, conv_inch, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(conv_inch, self.num_points, kernel_size=6)
        self.final_fc = nn.Linear(fc_inch, 3)
        self._ext = ext

    def forward(self, x):
        if self._ext:
            x = self.ReLU(self.conv1(x))
            x = self.ReLU(self.conv2(x))
        x = self.ReLU(self.final_conv(x))
        x = x.view(x.size(0), x.size(1), -1)
        x = self.final_fc(x)
        return x  # [8, 300, 3]


class segmentation_model_point(segmentation_models):
    def __init__(self, name='resnet50', pretrained=False, decoder_channels=(512, 256, 128, 64, 32), in_channel=3,
                 classes=4, multilvl=False, fc_inch=4, extpn=False):
        super(segmentation_model_point, self).__init__(name=name, pretrained=pretrained,
                                                       decoder_channels=decoder_channels, in_channel=in_channel,
                                                       classes=classes, multilvl=multilvl)
        self.pointnet = PointNet(num_points=300, fc_inch=fc_inch, conv_inch=2048, ext=extpn)
        print(f'Model {name} loaded.')
        print(f'Number of params: {get_n_params(self):,}')

    def forward(self, x, features_out=True):
        features = self.encoder(x)  # channels [3, 64, 256, 512, 1024, 2048]

        point = self.pointnet(features[-1])

        """decoder forward"""
        output = features[1:]  # remove first skip with same spatial resolution
        output = output[::-1]  # reverse channels to start from head of encoder

        head = output[0]
        skips = output[1:]

        decoder_output = self.decoder.center(head)
        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            target_height = skip.size(-2) if skip is not None else decoder_output.size(-2) * 2
            target_width = skip.size(-1) if skip is not None else decoder_output.size(-1) * 2
            decoder_output = decoder_block(decoder_output, target_height, target_width, skip_connection=skip)
            if i == len(self.decoder.blocks) - 2:
                output_aux = decoder_output
                output_aux = F.interpolate(output_aux, size=x.size()[2:], mode='bilinear', align_corners=True)
                output_aux = self.classifier_aux(output_aux)
        # decoder_output = self._decoder(*features)
        output = self.classifier(decoder_output)

        return output, output_aux, point


if __name__ == '__main__':
    from torch import rand
    from utils.utils_ import write_model_graph

    img = rand((2, 3, 224, 224))
    model = segmentation_models(name='resnet50', pretrained=False, decoder_channels=(512, 256, 128, 64, 32),
                                in_channel=3,
                                classes=4, multilvl=True)
    out = model(img)
    write_model_graph(model, img, '../runs/resnet50Mul')
