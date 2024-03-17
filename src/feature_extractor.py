import torch
import torch.nn as nn
import wandb

# for 84x84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def copy_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias
    # print("Are target and source pointing to the same reference? ")
    # print(id(trg.weight)==id(src.weight))


class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        assert size in {84, 100}, f"unexpected size: {size}"
        self.size = size

    def forward(self, x):
        assert x.ndim == 4, "input must be a 4D tensor"
        if x.size(2) == self.size and x.size(3) == self.size:
            return x
        assert x.size(3) == 100, f"unexpected size: {x.size(3)}"
        if self.size == 84:
            p = 8
        return x[:, :, p:-p, p:-p]


class NormalizeImg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.max() > 1:
            return x / 255.0


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class RLProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, x):
        """
        flatten -> Linear(100) + LayerNorm + Tanh
        """
        y = self.projection(x)
        return y


class SharedCNN(nn.Module):
    """Shared Convolutional encoder of pixels observations."""
    def __init__(self,
                 obs_shape,
                 num_layers,
                 num_filters):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.layers = [NormalizeImg(), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),]
        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers = nn.Sequential(*self.layers)

        self.out_dim = OUT_DIM_64[self.num_layers] if obs_shape[-1] == 64 else OUT_DIM[self.num_layers]

    def forward(self, x):
        """
        obs (中心裁剪/255.) -> conv + relu -> (conv + relu) * 9 -> conv
        """
        return self.layers(x)


class HeadCNN(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self,
                 in_dim,
                 num_layers,
                 num_filters):
        super().__init__()

        self.in_dim = in_dim
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)

        self.out_dim = in_dim
        self.apply(weight_init)

    def forward(self, x):
        """
        conv -> flatten
        """
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, shared_cnn, head_cnn, projection):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.head_cnn = head_cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.shared_cnn(x)
        x = self.head_cnn(x)
        if detach:
            x = x.detach()
        return self.projection(x)

class Discriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)


    def forward(self, obs):
        D_critic = torch.relu(self.ln(self.fc(obs)))
        D_critic = torch.tanh(self.head(D_critic))

        return D_critic


class Decoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(feature_dim, num_filters * self.out_dim * self.out_dim)

        self.deconvs = nn.ModuleList()

        for _ in range(self.num_layers - 1):
            self.deconvs.append(nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1))

        self.deconvs.append(nn.ConvTranspose2d(num_filters, obs_shape[0], 3, stride=2, output_padding=1))

        self.outputs = {}

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs["fc"] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs["deconv1"] = deconv

        for i in range(self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs["obs"] = obs

        return obs

    def log(self, L, step, log_freq, WB_LOG):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_decoder/conv%s' % (i + 1), self.deconvs[i], step)
        L.log_param('train_decoder/fc', self.fc, step)

        if WB_LOG:
            for i in range(self.num_layers):
                wandb.log({f"train_decoder/deconv{i + 1}": self.deconvs[i], "step": step})
            wandb.log({"train_decoder/fc": self.fc, "step": step})


AVAILABLE_ENCODERS = {"pixel": Encoder}
AVAILABLE_DECODERS = {"pixel": Decoder}


def make_encoder(encoder_type,
                 obs_shape,
                 feature_dim,
                 num_layers,
                 num_filters,
                 output_logits=False):
    assert encoder_type in AVAILABLE_ENCODERS
    return AVAILABLE_ENCODERS[encoder_type](obs_shape, feature_dim,
                                            num_layers, num_filters,
                                            output_logits)


def make_decoder(decoder_type,
                 obs_shape,
                 feature_dim,
                 num_layers,
                 num_filters):
    assert decoder_type in AVAILABLE_DECODERS
    return AVAILABLE_DECODERS[decoder_type](obs_shape, feature_dim,
                                            num_layers, num_filters)
