import torch
import torch.nn as nn
from einops import rearrange, reduce
from .layers import HerPNConv, Flatten, HerPNPool

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        if m.weight is not None:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()

class Backbone(nn.Module):
    def __init__(self, output_size):
        super(Backbone, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers = nn.Sequential(HerPNConv(16, 16),
                                    HerPNConv(16, 32, 2),
                                    HerPNConv(32, 32),
                                    HerPNConv(32, 64, 2),
                                    HerPNConv(64, 64))
        self.herpnpool = HerPNPool(64, output_size=output_size)
        self.flatten = Flatten()
        self.bn = nn.BatchNorm1d(output_size[0] * output_size[1] * 64)
    
    def forward(self, x):        
        out = self.conv(x)
        for m in self.layers:
            out = m(out)
        out = self.herpnpool(out)
        out = self.flatten(out)
        out = self.bn(out)
        return out
    
    def fuse(self):
        for m in self.layers:
            m.fuse()
        self.herpnpool.fuse()
        
    def forward_fuse(self, x):        
        out = self.conv(x)
        for m in self.layers:
            out = m.forward_fuse(out)
        out = self.herpnpool.forward_fuse(out)
        out = self.flatten(out)
        out = self.bn(out)
        return out


class PatchCNN(nn.Module):
    def __init__(self, input_size, patch_size):
        super(PatchCNN, self).__init__()
        
        self.H, self.W = input_size // patch_size, input_size // patch_size
        N = self.H * self.W
        output_size = (2, 2)
        dim = output_size[0] * output_size[1] * 64
        
        self.nets = nn.ModuleList([Backbone(output_size) for _ in range(N)])
        self.linear = nn.Linear(N * dim, 256)
        self.bn = nn.BatchNorm1d(256, affine=False)
        self.jigsaw = nn.Linear(dim, N)
            
        self.apply(initialize_weights)
    
    @torch.no_grad()
    def fuse(self):
        for m in self.nets:
            m.fuse()
        mean = self.bn.running_mean
        var = self.bn.running_var
        e = self.bn.eps 
        weight=self.linear.weight
        bias=self.linear.bias
        weight = torch.divide(weight.T, torch.sqrt(var + e))
        bias = torch.divide(bias - mean, torch.sqrt(var + e))
        weight = weight.T
        self.bias = nn.Parameter(bias / (self.H * self.W))
        self.weights = nn.ParameterList(torch.chunk(weight, self.H * self.W, dim=1))
    
    def _forward(self, x, fuse=False):
        H, W = self.H, self.W
        N = H * W
        x = rearrange(x, 'b c (h p1) (w p2) -> (h w) b c p1 p2', h=H, w=W)
        streams = [torch.cuda.Stream() for _ in range(N)]
        y = [None for _ in range(N)]
        for i in range(N):
            with torch.cuda.stream(streams[i]):
                if not fuse:
                    y[i] = self.nets[i](x[i])
                else:
                    y[i] = self.nets[i].forward_fuse(x[i])
                    y[i] = y[i] @ self.weights[i].T + self.bias
        torch.cuda.synchronize()
        out = torch.stack(y, dim=0)
        out = rearrange(out, 'n b c -> b n c')
        return out
    
    def forward_fuse(self, x):
        out = self._forward(x, True)
        out = reduce(out, 'b n c -> b c', 'sum')
        return out
    
    def forward(self, x):
        out = self._forward(x)
        out_global = out
        out_global = rearrange(out_global, 'b n c -> b (n c)')
        out_global = self.linear(out_global)
        out_global = self.bn(out_global)
        
        B, N, _ = out.shape
        pred = self.jigsaw(out)
        pred = pred.view(-1, N)
        target = torch.arange(0, N, device=out.device).repeat(B)
        
        return out_global, pred, target