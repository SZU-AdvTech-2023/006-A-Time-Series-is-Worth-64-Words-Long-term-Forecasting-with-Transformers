import os
import torch
from models import Autoformer,  DLinear, FEDformer, \
    LightTS, Reformer, ETSformer, Pyraformer, PatchTST,  PatchTST8, PatchTST32, PatchTST64, PatchTST_multi


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'PatchTST_multi':PatchTST_multi,
            'PatchTST64': PatchTST64,
            'PatchTST32': PatchTST32,
            'PatchTST8': PatchTST8
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
