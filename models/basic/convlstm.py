import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size, padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        self.conv = nn.Conv2d(
            in_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias)

    def init_hidden(self, batch_size, fmp_size):
        fmp_h, fmp_w = fmp_size
        h_init = torch.zeros(batch_size, self.hidden_dim, fmp_h, fmp_w, device=self.conv.weight.device)
        c_init = torch.zeros(batch_size, self.hidden_dim, fmp_h, fmp_w, device=self.conv.weight.device)

        return h_init, c_init


    def forward(self, x_cur, h_cur, c_cur):
        """
        Input:
            x_cur: (Tensor) [B, C_in, H, W]
            h_cur: (Tensor) [B, C_hd, H, W]
            c_cur: (Tensor) [B, C_hd, H, W]
        Output:

        """
        xh_cur = torch.cat([x_cur, h_cur], dim=1)
        # [B, C_in+C_hd, H, W] -> [B, 4*C_hd, H, W]
        y_cur = self.conv(xh_cur)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(y_cur, chunks=4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size, padding, dilation, num_layers,
                 bias=True, return_all_layers=False, inf_full_seq=True,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = [hidden_dim] * num_layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.inf_full_seq = inf_full_seq
        self.initialization = True

        assert len(self.hidden_dims) == num_layers

        cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                cur_in_dim = in_dim
            else:
                cur_in_dim = self.hidden_dims[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    in_dim=cur_in_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=self.bias)
                    )

        self.cell_list = nn.ModuleList(cell_list)


    def _init_hidden(self, batch_size, fmp_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, fmp_size))
        return init_states


    def inference_full_sequence(self, feats, hidden_state=None):
        """
        Input:
            feats: (List[Tensor]): List[T, B, C_in, H, W]
            hidden_state: 
        """
        seq_len = len(feats)
        B, _, H, W = feats[0].size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=B, fmp_size=(H, W))

        layer_output_list = []
        last_state_list = []

        x_cur = feats

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            outputs = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x_cur[t], h, c)
                outputs.append(h)

            x_cur = outputs

            layer_output_list.append(outputs)
            last_state_list.append([h, c])

        if self.return_all_layers:
            return layer_output_list, last_state_list

        else:
            return layer_output_list[-1:], last_state_list[-1:]


    def inference_new_input(self, feat, last_hidden_states):
        """
        Input:
            feat: (Tensor) [B, C, H, W]
            hidden_state:
        """
        x_cur = feat
        layer_output_list = []
        last_state_list = []

        for layer_idx in range(self.num_layers):
            h, c = last_hidden_states[layer_idx]

            # infer cur input
            h, c = self.cell_list[layer_idx](x_cur, h, c)
            x_cur = h

            layer_output_list.append(h)
            last_state_list.append([h, c])

        if self.return_all_layers:
            return layer_output_list, last_state_list

        else:
            return layer_output_list[-1:], last_state_list[-1:]


    def forward(self, feat, hidden_state=None):
        if self.inf_full_seq:
            return self.inference_full_sequence(feat, hidden_state)
        else:
            if self.initialization:
                self.return_all_layers = True
                (
                    layer_output_list,
                    last_state_list
                ) = self.inference_full_sequence(feat, hidden_state)
                self.initialization = False
                self.last_state_list = last_state_list
                
            else:
                # cur_input = feats[-1]
                (
                    layer_output_list,
                    last_state_list
                ) = self.inference_new_input(feat, self.last_state_list)
                self.last_state_list = last_state_list


            return layer_output_list, last_state_list


# build ConvLSTM
def build_convlstm(cfg, in_dim):
    model = ConvLSTM(
        in_dim=in_dim,
        hidden_dim=cfg['head_dim'],
        kernel_size=cfg['conv_lstm_ks'],
        padding=cfg['conv_lstm_pd'],
        dilation=cfg['conv_lstm_di'],
        num_layers=cfg['conv_lstm_nl'],
        return_all_layers=False,
        inf_full_seq=True
    )

    return model


if __name__ == '__main__':
    feats = [torch.randn(2, 16, 10, 10) for _ in range(8)]

    inf_full_seq = True

    convlstms = ConvLSTM(in_dim=16,
                         hidden_dims=[8, 8],
                         kernel_size=3,
                         num_layers=2,
                         bias=True,
                         return_all_layers=False,
                         inf_full_seq=inf_full_seq
                         )

    if inf_full_seq:
        outputs = convlstms(feats)
        print(outputs[0][-1][-1].shape)
    else:
        outputs = convlstms(feats)
        output = convlstms(feats)

