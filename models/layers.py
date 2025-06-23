import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class HerPNConv(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(HerPNConv, self).__init__()
        self.herpn1 = HerPN(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.herpn2 = HerPN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.planes = planes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes))

    def forward(self, x):
        x = self.herpn1(x)
        out = self.conv1(x)
        out = self.herpn2(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out
    
    def forward_fuse(self, x):
        x = torch.square(x) + self.a1 * x + self.a0
        out = F.conv2d(x, self.weight1, stride=self.conv1.stride, padding=self.conv1.padding)
        out = torch.square(out) + self.b1 * out + self.b0
        out = F.conv2d(out, self.weight2, stride=self.conv2.stride, padding=self.conv2.padding)
        out += self.shortcut(x * self.a2)
        return out
    
    @torch.no_grad()
    def fuse(self):
        m0, v0 = self.herpn1.bn0.running_mean, self.herpn1.bn0.running_var
        m1, v1 = self.herpn1.bn1.running_mean, self.herpn1.bn1.running_var
        m2, v2 = self.herpn1.bn2.running_mean, self.herpn1.bn2.running_var
        g, b = self.herpn1.weight.squeeze(), self.herpn1.bias.squeeze()
        e = self.herpn1.bn0.eps
        a2 = torch.divide(g, torch.sqrt(8 * math.pi * (v2 + e)))
        a1 = torch.divide(g, 2 * torch.sqrt(v1 + e))
        a0 = b + g * (torch.divide(1 - m0, torch.sqrt(2 * math.pi * (v0 + e))) - torch.divide(m1, 2 * torch.sqrt(v1 + e)) - torch.divide(1 + math.sqrt(2) * m2, torch.sqrt(8 * math.pi * (v2 + e))))
        a2 = a2.unsqueeze_(-1).unsqueeze_(-1)
        a1 = a1.unsqueeze_(-1).unsqueeze_(-1)
        a0 = a0.unsqueeze_(-1).unsqueeze_(-1)

        m0, v0 = self.herpn2.bn0.running_mean, self.herpn2.bn0.running_var
        m1, v1 = self.herpn2.bn1.running_mean, self.herpn2.bn1.running_var
        m2, v2 = self.herpn2.bn2.running_mean, self.herpn2.bn2.running_var
        g, b = self.herpn2.weight.squeeze(), self.herpn2.bias.squeeze()
        e = self.herpn2.bn0.eps
        b2 = torch.divide(g, torch.sqrt(8 * math.pi * (v2 + e)))
        b1 = torch.divide(g, 2 * torch.sqrt(v1 + e))
        b0 = b + g * (torch.divide(1 - m0, torch.sqrt(2 * math.pi * (v0 + e))) - torch.divide(m1, 2 * torch.sqrt(v1 + e)) - torch.divide(1 + math.sqrt(2) * m2, torch.sqrt(8 * math.pi * (v2 + e))))
        b2 = b2.unsqueeze_(-1).unsqueeze_(-1)
        b1 = b1.unsqueeze_(-1).unsqueeze_(-1)
        b0 = b0.unsqueeze_(-1).unsqueeze_(-1)
        
        weight1 = self.conv1.weight * a2
        weight2 = self.conv2.weight * b2
        a1 = a1 / a2
        a0 = a0 / a2
        b1 = b1 / b2
        b0 = b0 / b2
        
        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.a2 = nn.Parameter(a2)
        self.a1 = nn.Parameter(a1)
        self.a0 = nn.Parameter(a0)
        self.b1 = nn.Parameter(b1)
        self.b0 = nn.Parameter(b0)
    
    
class HerPNPool(nn.Module):
    def __init__(self, planes, output_size):
        super(HerPNPool, self).__init__()
        self.herpn = HerPN(planes)
        self.pool = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, x):
        x = self.herpn(x)
        out = self.pool(x)
        return out
    
    @torch.no_grad()
    def fuse(self):
        m0, v0 = self.herpn.bn0.running_mean, self.herpn.bn0.running_var
        m1, v1 = self.herpn.bn1.running_mean, self.herpn.bn1.running_var
        m2, v2 = self.herpn.bn2.running_mean, self.herpn.bn2.running_var
        g, b = self.herpn.weight.squeeze(), self.herpn.bias.squeeze()
        e = self.herpn.bn0.eps
        a2 = torch.divide(g, torch.sqrt(8 * math.pi * (v2 + e)))
        a1 = torch.divide(g, 2 * torch.sqrt(v1 + e))
        a0 = b + g * (torch.divide(1 - m0, torch.sqrt(2 * math.pi * (v0 + e))) - torch.divide(m1, 2 * torch.sqrt(v1 + e)) - torch.divide(1 + math.sqrt(2) * m2, torch.sqrt(8 * math.pi * (v2 + e))))
        a2 = a2.unsqueeze_(-1).unsqueeze_(-1)
        a1 = a1.unsqueeze_(-1).unsqueeze_(-1)
        a0 = a0.unsqueeze_(-1).unsqueeze_(-1)
        a1 = a1 / a2
        a0 = a0 / a2
        self.a2 = nn.Parameter(a2)
        self.a1 = nn.Parameter(a1)
        self.a0 = nn.Parameter(a0)
        
    def forward_fuse(self, x):
        out = torch.square(x) + self.a1 * x + self.a0
        out = F.adaptive_avg_pool2d(out, self.pool.output_size) * self.a2
        return out
        

class HerPN(nn.Module):
    def __init__(self, planes):
        super(HerPN, self).__init__()
        self.bn0 = nn.BatchNorm2d(planes, affine=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)
        self.weight = nn.Parameter(torch.ones(planes, 1, 1))
        self.bias = nn.Parameter(torch.zeros(planes, 1, 1))

    def forward(self, x):
        x0 = self.bn0(torch.ones_like(x))
        x1 = self.bn1(x)
        x2 = self.bn2((torch.square(x) - 1) / math.sqrt(2))
        out = torch.divide(x0, math.sqrt(2 * math.pi)) + torch.divide(x1, 2) + torch.divide(x2, np.sqrt(4 * math.pi))
        out = self.weight * out + self.bias
        return out
    

class HerPN_Fuse(nn.Module):
    def __init__(self, herpn: HerPN):
        super(HerPN_Fuse, self).__init__()
        with torch.no_grad():
            m0, v0 = herpn.bn0.running_mean, herpn.bn0.running_var
            m1, v1 = herpn.bn1.running_mean, herpn.bn1.running_var
            m2, v2 = herpn.bn2.running_mean, herpn.bn2.running_var
            g, b = herpn.weight.squeeze(), herpn.bias.squeeze()
            e = herpn.bn0.eps
            w2 = torch.divide(g, torch.sqrt(8 * math.pi * (v2 + e)))
            w1 = torch.divide(g, 2 * torch.sqrt(v1 + e))
            w0 = b + g * (torch.divide(1 - m0, torch.sqrt(2 * math.pi * (v0 + e))) - torch.divide(m1, 2 * torch.sqrt(v1 + e)) - torch.divide(1 + math.sqrt(2) * m2, torch.sqrt(8 * math.pi * (v2 + e))))
        self.w2 = nn.Parameter(w2.unsqueeze_(-1).unsqueeze_(-1))
        self.w1 = nn.Parameter(w1.unsqueeze_(-1).unsqueeze_(-1))
        self.w0 = nn.Parameter(w0.unsqueeze_(-1).unsqueeze_(-1))

    def forward(self, x):
        out = self.w2 * torch.square(x) + self.w1 * x + self.w0
        return out