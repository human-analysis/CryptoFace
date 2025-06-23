import math
import torch
import torch.nn as nn

class ArcFace(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, s: float = 64., m: float = 0.5):
        super(ArcFace, self).__init__()
        self.kernel = nn.Parameter(torch.Tensor(embedding_dim, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.num_classes = num_classes
        self.m = m
        self.s = s
        self.eps = 1e-4

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        :param embeddings: not normalized embeddings
        :param labels: integer labels
        :return:
        """
        embeddings_norm = torch.norm(embeddings, 2, 1, True)
        embeddings = torch.divide(embeddings, embeddings_norm)
        
        kernel_norm = torch.norm(self.kernel, 2, 0, True)
        kernel = torch.divide(self.kernel, kernel_norm)

        cosine = embeddings @ kernel
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)

        m_hot = torch.zeros(labels.size()[0], cosine.size()[1])
        m_hot = m_hot.to(cosine)
        m_hot.scatter_(1, labels.reshape(-1, 1), self.m)

        theta = cosine.acos()
        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi - self.eps)
        cosine_m = theta_m.cos()
        scaled_cosine_m = cosine_m * self.s

        return scaled_cosine_m
    