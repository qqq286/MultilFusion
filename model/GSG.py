import torch
from torch import nn
from torch.nn import functional as F
import math

class AdaptiveGaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, initial_sigma=1.0, max_sigma=5.0):
        super(AdaptiveGaussianFilter, self).__init__()
        self.kernel_size = kernel_size  # Size of the Gaussian kernel
        self.padding = kernel_size // 2  # Padding to keep the output size same as input
        
        # Learnable parameter sigma (initialized to the given value)
        self.sigma = nn.Parameter(torch.tensor(initial_sigma, dtype=torch.float32))
        self.max_sigma = max_sigma

    def forward(self, weights):
        """
        weights: input of shape (batch_size, num_frames, 1), where num_frames is the number of time steps (e.g., 32).
        """
        # Generate the Gaussian kernel dynamically based on sigma
        kernel = self.create_gaussian_kernel(self.kernel_size, self.sigma, device=weights.device)
        
        # Apply Gaussian smoothing using 1D convolution (along the frame dimension)
        weights = weights.permute(0, 2, 1)  # [bs, 1, num_frames]
        smoothed_weights = F.conv1d(weights, kernel, padding=self.padding)
        smoothed_weights = smoothed_weights.permute(0, 2, 1)  # [bs, num_frames, 1]
        # Re-normalize the weights (like Softmax) so that they sum to 1 along the time dimension
        smoothed_weights = F.softmax(smoothed_weights, dim=1)
        return smoothed_weights  # Return with original shape

    def create_gaussian_kernel(self, kernel_size, sigma, device):
        # Create a 1D tensor from -kernel_size//2 to kernel_size//2
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        # Compute the Gaussian kernel
        gaussian_kernel = torch.exp(-0.5 * (x / sigma).pow(2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # Reshape to (out_channels, in_channels, kernel_size) for conv1d
        return gaussian_kernel.view(1, 1, -1)

    def create_gaussian_kernel_v2(self, kernel_size, gate, device):
        sigma = self.sigma(gate.squeeze(dim=-1)).unsqueeze(dim=-1) * self.max_sigma # [bs, 1, 1]
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        x = x.view(1, 1, kernel_size)
        gaussian_kernels = torch.exp(-0.5 * (x / sigma).pow(2))
        gaussian_kernels = gaussian_kernels / gaussian_kernels.sum(dim=-1, keepdim=True)
        return gaussian_kernels  # shape: (bs, 1, kernel_size)

class GroundingModule(nn.Module):
    def __init__(self, d_model=768, dropout=0.3):
        super().__init__()
        self.qa_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
    
        self.v_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        
        self.grounding = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=-2)
        )

        self.time_estimate = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.Linear(d_model // 2, 3),
            nn.ReLU()
        )
    
        self.gs_filter = AdaptiveGaussianFilter()
        # 为后续 ground_v2 使用位置编码，这里直接调用生成函数
        self.pos_embedding = self._get_pos_embedding(max_len=32, d_model=d_model)
    
    def _get_pos_embedding(self, max_len=32, d_model=768):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        return pe

    def gen_grounding_v2(self, keyframe_prob, v, alpha=20):
        bs, length = v.size()[:2]
        activated_v = torch.sum(keyframe_prob.unsqueeze(dim=-1) * v, dim=1)  # [bs, d_model]
        time_output = self.time_estimate(activated_v)
        keyframe = time_output[:, 0]
        start = time_output[:, 1]
        end = time_output[:, 2]
        start_time = (keyframe - start).sigmoid()
        end_time = (keyframe + end).sigmoid()
        # 生成 mask
        positions = torch.linspace(0, 1, length, device=v.device)
        start_weights = torch.sigmoid(alpha * (positions - start_time.unsqueeze(1)))
        end_weights = torch.sigmoid(alpha * (end_time.unsqueeze(1) - positions))
        keyframe_mask = start_weights * end_weights  # [bs, length]
        pos_embedding = keyframe_mask.unsqueeze(dim=-1) * self.pos_embedding[:, :length, :]
        time_param = {
            "key": keyframe_prob, 
            "start": start_time, 
            "end": end_time, 
            "mask": keyframe_mask, 
            "pos_embed": pos_embedding
        }
        return time_param
    
    def gen_grounding(self, keyframe_prob, window_sizes=[1, 3, 5]):
        bs, length = keyframe_prob.size()[:2]
        max_indices = self.find_best_interval_v2(keyframe_prob)
        start_indices = max_indices[:, 0].unsqueeze(1).expand(-1, length)
        end_indices = max_indices[:, 1].unsqueeze(1).expand(-1, length)
        range_tensor = torch.arange(length, device=keyframe_prob.device).expand(bs, length)
        keyframe_mask = (range_tensor >= start_indices) & (range_tensor <= end_indices)
        start_time = max_indices[:, 0] / (length - 1)
        end_time = max_indices[:, 1] / (length - 1)
        time_param = {
            "key": keyframe_prob, 
            "max_indices": max_indices, 
            "start": start_time, 
            "end": end_time, 
            "mask": keyframe_mask
        }
        return time_param
    
    def find_best_interval_v2(self, keyframe_prob, window_sizes=[1, 3, 5]):
        bs, length = keyframe_prob.shape
        max_interval_size = length // 2
        max_indices = torch.zeros((bs, 2), dtype=torch.long, device=keyframe_prob.device)
        best_scores = torch.full((bs,), float('-inf'), dtype=torch.float, device=keyframe_prob.device)
        
        for window_size in window_sizes:
            if window_size > max_interval_size:
                continue
            sliding_sums = F.conv1d(
                keyframe_prob.unsqueeze(1), 
                weight=torch.ones((1, 1, window_size), device=keyframe_prob.device), 
                padding=0, 
                stride=1
            ).squeeze(1)
            max_positions = keyframe_prob.argmax(dim=1, keepdim=True)
            for start in range(length - window_size + 1):
                end = start + window_size
                contains_max = (max_positions >= start) & (max_positions < end)
                window_scores = sliding_sums[:, start]
                window_scores[~contains_max.squeeze()] = float('-inf')
                better_scores = window_scores > best_scores
                best_scores = torch.where(better_scores, window_scores, best_scores)
                max_indices[better_scores] = torch.tensor([start, end], device=keyframe_prob.device)
        return max_indices
    
    def _sample_negatives(self, x_pos, k):
        bs, n = x_pos.size()
        x_neg = torch.zeros(bs, k, n).to(x_pos.device)
        for i in range(bs):
            indices = list(range(bs))
            indices.remove(i)
            neg_indices = torch.tensor(indices).to(x_pos.device)
            sampled_indices = neg_indices[torch.randint(0, len(neg_indices), (k,))]
            x_neg[i] = x_pos[sampled_indices]
        return x_neg
    
    def time_penalty(self, time_param, lambda_1=1, lambda_2=1):
        t_start = time_param["start"]
        t_end = time_param["end"]
        probs = time_param["key"]
        frame_count = probs.size(1)
        frame_positions = torch.linspace(0, 1, steps=frame_count).to(t_start.device)
        t_start = t_start.unsqueeze(1).expand(-1, frame_count)
        t_end = t_end.unsqueeze(1).expand(-1, frame_count)
        conf_weights = ((frame_positions < t_start) | (frame_positions > t_end)).float()
        keyframe_neg = (probs * conf_weights).sum(dim=1)
        keyframe_pos = (probs * (1 - conf_weights)).sum(dim=1)
        keyframe_penalty_mean = (keyframe_neg - keyframe_pos).mean()
        time_penalty = keyframe_penalty_mean
        time_param["time_penalty"] = time_penalty
        return time_param
    
    def forward(self, v, qa):
        # v: [bs, length, d_model]; qa: [bs, d_model]
        v = self.v_proj(v)
        qa = self.qa_proj(qa)
        gate = torch.matmul(v, qa.unsqueeze(dim=-1)).tanh()  # [bs, length, 1]
        keyframe_prob = self.grounding(v * gate)  # [bs, length, 1]
        keyframe_prob_gs = self.gs_filter(keyframe_prob)  # [bs, length, 1]
        keyframe_prob_gs = keyframe_prob_gs.squeeze(dim=-1)
        time_param = self.gen_grounding(keyframe_prob_gs)
        time_param["ori_key"] = keyframe_prob.squeeze(dim=-1)
        return time_param

    def forward_v2(self, v, qa):
        v = self.v_proj(v)
        qa = self.qa_proj(qa)
        keyframe_prob = torch.matmul(v, qa.unsqueeze(dim=-1)).softmax(dim=1)
        keyframe_prob = self.grounding(v * gate)  # Note: gate is undefined here; using forward_v2 requires进一步修改.
        keyframe_prob = self.gs_filter(keyframe_prob.squeeze(dim=-1))
        time_param = self.gen_grounding_v2(keyframe_prob, v)
        time_param["neg_key"] = self._sample_negatives(time_param["mask"].float(), v.size(0) // 8)
        return time_param

if __name__ == '__main__':
    # 使用 B, N, C 表示输入维度
    B, N, C = 2, 32, 768  # B: batch_size, N: 序列长度, C: 特征维度
    input_v = torch.randn(B, N, C)    # 视频特征输入
    input_qa = torch.randn(B, C)      # 问题文本输入

    model = GroundingModule(d_model=C)
    # 设置位置编码，生成 [1, N, C] 的位置编码
    model.pos_embedding = model._get_pos_embedding(max_len=N, d_model=C)

    output = model(input_v, input_qa)
    print("input_v:", input_v.shape)
    print("input_qa:", input_qa.shape)

