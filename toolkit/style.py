from torch import nn
import torch.nn.functional as F
import torch
from torchvision import models


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tensor_size(tensor):
    channels = tensor.shape[1]
    height = tensor.shape[2]
    width = tensor.shape[3]
    return channels * height * width

class ContentLoss(nn.Module):

    def __init__(self, single_target=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(ContentLoss, self).__init__()
        self.single_target = single_target
        self.device = device
        self.loss = None

    def forward(self, stacked_input):

        if self.single_target:
            split_size = stacked_input.size()[0] // 2
            pred_layer, target_layer = torch.split(stacked_input, split_size, dim=0)
        else:
            split_size = stacked_input.size()[0] // 3
            pred_layer, _, target_layer = torch.split(stacked_input, split_size, dim=0)

        content_size = tensor_size(pred_layer)

        # Define the separate loss function
        def separated_loss(y_pred, y_true):
            y_pred = y_pred.float()
            y_true = y_true.float()
            diff = torch.abs(y_pred - y_true)
            l2 = torch.sum(diff ** 2, dim=[1, 2, 3], keepdim=True) / 2.0
            return 2. * l2 / content_size

        # Calculate itemized loss
        pred_itemized_loss = separated_loss(pred_layer, target_layer)
        # check if is nan
        if torch.isnan(pred_itemized_loss).any():
            print('pred_itemized_loss is nan')

        # Calculate the mean of itemized loss
        loss = torch.mean(pred_itemized_loss, dim=(1, 2, 3), keepdim=True)
        self.loss = loss

        return stacked_input


def convert_to_gram_matrix(inputs):
    inputs = inputs.float()
    shape = inputs.size()
    batch, filters, height, width = shape[0], shape[1], shape[2], shape[3]
    size = height * width * filters

    feats = inputs.view(batch, filters, height * width)
    feats_t = feats.transpose(1, 2)
    grams_raw = torch.matmul(feats, feats_t)
    gram_matrix = grams_raw / size

    return gram_matrix


######################################################################
# Now the style loss module looks almost exactly like the content loss
# module. The style distance is also computed using the mean square
# error between :math:`G_{XL}` and :math:`G_{SL}`.
#

class StyleLoss(nn.Module):

    def __init__(self, single_target=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(StyleLoss, self).__init__()
        self.single_target = single_target
        self.device = device

    def forward(self, stacked_input):
        input_dtype = stacked_input.dtype
        stacked_input = stacked_input.float()
        if self.single_target:
            split_size = stacked_input.size()[0] // 2
            preds, style_target = torch.split(stacked_input, split_size, dim=0)
        else:
            split_size = stacked_input.size()[0] // 3
            preds, style_target, _ = torch.split(stacked_input, split_size, dim=0)

        def separated_loss(y_pred, y_true):
            gram_size = y_true.size(1) * y_true.size(2)
            sum_axis = (1, 2)
            diff = torch.abs(y_pred - y_true)
            raw_loss = torch.sum(diff ** 2, dim=sum_axis, keepdim=True)
            return raw_loss / gram_size

        target_grams = convert_to_gram_matrix(style_target)
        pred_grams = convert_to_gram_matrix(preds)
        itemized_loss = separated_loss(pred_grams, target_grams)
        # check if is nan
        if torch.isnan(itemized_loss).any():
            print('itemized_loss is nan')
        # reshape itemized loss to be (batch, 1, 1, 1)
        itemized_loss = torch.unsqueeze(itemized_loss, dim=1)
        # gram_size = (tf.shape(target_grams)[1] * tf.shape(target_grams)[2])
        loss = torch.mean(itemized_loss, dim=(1, 2), keepdim=True)
        self.loss = loss.to(input_dtype).float()
        return stacked_input.to(input_dtype)


# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(Normalization, self).__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.dtype = dtype
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, stacked_input):
        # cast to float 32 if not already # only necessary when processing gram matrix
        # if stacked_input.dtype != torch.float32:
        #     stacked_input = stacked_input.float()
        # remove alpha channel if it exists
        if stacked_input.shape[1] == 4:
            stacked_input = stacked_input[:, :3, :, :]
        # normalize to min and max of 0 - 1
        in_min = torch.min(stacked_input)
        in_max = torch.max(stacked_input)
        # norm_stacked_input = (stacked_input - in_min) / (in_max - in_min)
        # return (norm_stacked_input - self.mean) / self.std
        return ((stacked_input - self.mean) / self.std).to(self.dtype)


class OutputLayer(nn.Module):
    def __init__(self, name='output_layer'):
        super(OutputLayer, self).__init__()
        self.name = name
        self.tensor = None

    def forward(self, stacked_input):
        self.tensor = stacked_input
        return stacked_input


def get_style_model_and_losses(
        single_target=True,  # false has 3 targets, dont remember why i added this initially, this is old code
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_layer_name=None,
        dtype=torch.float32
):
    # content_layers = ['conv_4']
    # style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    content_layers = ['conv2_2', 'conv3_2', 'conv4_2']
    style_layers = ['conv2_1', 'conv3_1', 'conv4_1']
    cnn = models.vgg19(pretrained=True).features.to(device, dtype=dtype).eval()
    # set all weights in the model to our dtype
    # for layer in cnn.children():
    #     layer.to(dtype=dtype)

    # normalization module
    normalization = Normalization(device, dtype=dtype).to(device)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    block = 1
    children = list(cnn.children())

    output_layer = None

    for layer in children:
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv{block}_{i}_raw'
        elif isinstance(layer, nn.ReLU):
            # name = 'relu_{}'.format(i)
            name = f'conv{block}_{i}'  # target this
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            block += 1
            i = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            content_loss = ContentLoss(single_target=single_target, device=device)
            model.add_module("content_loss_{}_{}".format(block, i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            style_loss = StyleLoss(single_target=single_target, device=device)
            model.add_module("style_loss_{}_{}".format(block, i), style_loss)
            style_losses.append(style_loss)

        if output_layer_name is not None and name == output_layer_name:
            output_layer = OutputLayer(name)
            model.add_module("output_layer_{}_{}".format(block, i), output_layer)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) or isinstance(model[i], OutputLayer):
            break

    model = model[:(i + 1)]
    model.to(dtype=dtype)

    return model, style_losses, content_losses, output_layer
