import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import math
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# 设置中文字体支持
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


class GradCAM:
    """
    实现Grad-CAM算法，支持处理MAE等Transformer模型的序列特征
    """

    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None

        # 注册钩子
        self.hook_handles = []

        # 正向钩子：保存特征图
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        # 反向钩子：保存梯度
        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0].detach()

        # 获取目标层
        target_found = False
        for name, module in self.model.named_modules():
            if name == target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))
                target_found = True
                break

        if not target_found:
            raise ValueError(f"无法找到目标层: {target_layer}")

    def __call__(self, input_tensor, class_idx=None):
        # 设置为评估模式
        self.model.eval()

        # 前向传播
        self.model.zero_grad()
        output = self.model(input_tensor)

        # 如果未指定类别索引，则使用预测的最高概率类别
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # 反向传播
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 处理梯度和特征图
        if len(self.gradient.shape) == 4:  # 传统CNN格式 [B, C, H, W]
            # 对空间维度(H,W)求平均得到通道权重
            weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
            # 加权组合特征图
            cam = torch.sum(weights * self.feature_maps, dim=1).squeeze()

        elif len(self.gradient.shape) == 3:  # Transformer序列格式 [B, seq_len, C]
            # 移除CLS token（如果存在）
            if self.feature_maps.size(1) > 196:  # 14x14=196个patch (MAE通常有197=196+1)
                features = self.feature_maps[:, 1:, :]  # 移除CLS token
                grads = self.gradient[:, 1:, :]
            else:
                features = self.feature_maps
                grads = self.gradient

            # 对序列维度求平均得到通道权重
            weights = torch.mean(grads, dim=1, keepdim=True)  # [B, 1, C]
            # 加权组合特征图
            cam = torch.sum(weights * features, dim=2).squeeze()  # [seq_len]

            # 计算patch大小 (通常为14x14=196)
            patch_size = int(math.sqrt(features.size(1)))

            # 确保序列长度是一个完全平方数
            if patch_size * patch_size == features.size(1):
                # 重塑为二维热力图
                cam = cam.reshape(patch_size, patch_size)
                # 插值到输入图像大小
                cam = F.interpolate(
                    cam.unsqueeze(0).unsqueeze(0),  # 添加batch和channel维度
                    size=(input_tensor.size(2), input_tensor.size(3)),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()  # 移除batch和channel维度
            else:
                raise ValueError(f"序列长度({features.size(1)})不是一个完全平方数，无法重塑为二维热力图")
        else:
            raise ValueError(f"不支持的特征图维度: {self.feature_maps.shape}")

        # ReLU激活，因为我们只关心积极影响
        cam = torch.clamp(cam, min=0)

        # 归一化
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)

        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        """移除注册的钩子"""
        for handle in self.hook_handles:
            handle.remove()


def visualize_gradcam(
        image_path: str,
        model: nn.Module,
        target_layer: str,
        class_idx: Optional[int] = None,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        save_path: Optional[str] = None,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    可视化Grad-CAM结果

    Args:
        image_path: 输入图像路径
        model: 用于Grad-CAM的模型
        target_layer: 目标层名称
        class_idx: 目标类别索引
        question: 问题文本
        answer: 答案文本
        save_path: 保存路径，如果为None则显示图像
        alpha: 热力图透明度
        colormap: 热力图颜色映射

    Returns:
        superimposed_img: 叠加了热力图的图像
    """
    # print(model)

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    orig_size = image.size

    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    # 初始化GradCAM
    grad_cam = GradCAM(model, target_layer)

    # 计算CAM
    cam = grad_cam(input_tensor, class_idx)

    # 释放钩子
    grad_cam.remove_hooks()

    # 调整CAM大小为原始图像尺寸
    cam = cv2.resize(cam, orig_size)

    # 将CAM转换为热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 将热力图叠加到原始图像上
    img_np = np.array(image)
    superimposed_img = heatmap * alpha + img_np
    superimposed_img = np.uint8(superimposed_img)

    # 显示结果
    plt.figure(figsize=(12, 8))

    # 显示原始图像
    # plt.subplot(1, 2, 1)
    # plt.imshow(img_np)
    # # title = "原始图像"
    # # if question:
    # #     title += f"\n问题: {question}"
    # # plt.title(title)
    # plt.axis('off')

    # 显示Grad-CAM热力图
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    title = "Grad-CAM热力图"
    # if answer:
    #     title += f"\n答案: {answer}"
    #     print('answer',title)
    # plt.title(title)
    print('answer',title)
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

    plt.show()

    return superimposed_img


class VisionEncoderWrapper(nn.Module):
    """
    包装视觉编码器，使其可以直接处理图像输入并输出分类结果
    """

    def __init__(self, vision_encoder: nn.Module, classifier: nn.Module, pooling_type: str = "avg"):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.classifier = classifier
        self.pooling_type = pooling_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提取视觉特征
        features = self.vision_encoder(x)

        if isinstance(features, dict):
            features = features['x']  # 根据实际情况调整

        # 根据特征形状进行适当的池化
        if len(features.shape) == 4:  # [B, C, H, W]
            if self.pooling_type == "avg":
                pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
            elif self.pooling_type == "max":
                pooled_features = F.adaptive_max_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
            else:
                raise ValueError(f"不支持的池化类型: {self.pooling_type}")
        elif len(features.shape) == 3:  # [B, seq_len, C] (如Transformer输出)
            # 通常使用[CLS]标记或平均池化
            pooled_features = features.mean(dim=1)
        else:
            raise ValueError(f"不支持的特征形状: {features.shape}")

        # 通过分类器获取输出
        output = self.classifier(pooled_features)
        return output


def get_vqa_model_for_gradcam(
        model: nn.Module,
        vision_encoder_attr: str = "visual_encoder",
        classifier_attr: str = "classifier",
        pooling_type: str = "avg"
) -> nn.Module:
    """
    从VQA模型中提取用于Grad-CAM的视觉编码器模型

    Args:
        model: 完整的VQA模型
        vision_encoder_attr: 视觉编码器的属性名称
        classifier_attr: 分类器的属性名称
        pooling_type: 特征池化类型

    Returns:
        wrapped_model: 包装后的模型，适用于Grad-CAM
    """
    # 提取视觉编码器
    # print('11111111111',model)
    # print('2222222222',vision_encoder_attr)

    vision_encoder = getattr(model, vision_encoder_attr, None)
    if vision_encoder is None:
        raise ValueError(f"无法在模型中找到视觉编码器: {vision_encoder_attr}")

    # 提取分类器
    classifier = getattr(model, classifier_attr, None)
    if classifier is None:
        # 尝试从模型结构中推断分类器
        if hasattr(model, "module"):  # 对于DataParallel包装的模型
            classifier = model.module.classifier
        else:
            raise ValueError(f"无法在模型中找到分类器: {classifier_attr}")

    # 创建用于Grad-CAM的包装模型
    wrapped_model = VisionEncoderWrapper(vision_encoder, classifier, pooling_type)
    wrapped_model.to(next(model.parameters()).device)
    wrapped_model.eval()

    return wrapped_model


from ruamel.yaml import YAML


def main():
    """主函数：演示如何使用Grad-CAM可视化VQA模型"""
    import argparse

    parser = argparse.ArgumentParser(description='VQA模型的Grad-CAM可视化')
    parser.add_argument('--image_path', default='D:/Fusion/data/vqa/data_tcm/images/image_00002.jpg',
                        type=str, required=True, help='输入图像路径')
    parser.add_argument('--target_layer', default='visual_encoder.blocks.2.mlp', type=str, required=True,
                        help='目标层名称')
    parser.add_argument('--question', type=str, default='', help='问题文本')
    parser.add_argument('--answer_idx', type=int, default=None, help='答案类别索引')
    parser.add_argument('--save_path', type=str, default='D:/Fusion/results/grad_cam_result.png',
                        help='保存结果的路径')
    parser.add_argument('--vision_encoder_attr', type=str, default="visual_encoder", help='视觉编码器属性名')
    parser.add_argument('--classifier_attr', type=str, default="itm_head", help='分类器属性名')
    parser.add_argument('--pooling_type', type=str, default="avg", choices=["avg", "max"], help='特征池化类型')

    parser.add_argument('--config', default='D:/Fusion/configs/VQA.yaml')
    parser.add_argument('--output_dir', default='D:/Fusion/output/visual')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--dataset_use', default='rad', help='choose medical vqa dataset(rad, pathvqa, slake)')
    parser.add_argument('--checkpoint', default='D:/Fusion/output/vqa/rad/med_pretrain_49_rad_25.pth')
    parser.add_argument('--output_suffix', default='', help='output suffix, eg. ../rad_29_1')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='D:/Fusion/weights/pre_training/bert-base-uncased')
    parser.add_argument('--text_decoder', default='D:/Fusion/weights/pre_training/bert-base-uncased')
    args = parser.parse_args()

    yaml = YAML(typ='safe', pure=True)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型（根据您的实际模型修改）
    print(f"加载模型: {args.checkpoint}")

    # 尝试加载模型权重
    checkpoint = torch.load(args.checkpoint, map_location=device)
 

    # 检查checkpoint类型
    if isinstance(checkpoint, dict):
        # 从checkpoint中提取模型权重
        if 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            # 假设checkpoint直接是模型权重
            model_state_dict = checkpoint

        # 初始化模型（需要根据您的实际模型类进行修改）
        # 导入您的模型类
        from qmodel.model_vqa import Fusion_VQA
        from qmodel.tokenization_bert import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
        model = Fusion_VQA(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)

        # 加载模型权重
        model.load_state_dict(model_state_dict)
    else:
        # 如果checkpoint已经是模型对象
        model = checkpoint

    # 加载模型后
    # print_model_layers(model)

    model.to(device)
    model.eval()

    # 创建用于Grad-CAM的模型
    grad_cam_model = get_vqa_model_for_gradcam(
        model,
        vision_encoder_attr=args.vision_encoder_attr,
        classifier_attr=args.classifier_attr,
        pooling_type=args.pooling_type
    )

    # 获取答案映射（根据您的实际情况修改）
    answer_to_idx = None
    if hasattr(model, 'answer_to_idx'):
        answer_to_idx = model.answer_to_idx
    elif hasattr(model, 'module') and hasattr(model.module, 'answer_to_idx'):
        answer_to_idx = model.module.answer_to_idx

    # 准备答案文本
    answer_text = None
    if args.answer_idx is not None and answer_to_idx is not None:
        # 从索引获取答案文本
        idx_to_answer = {v: k for k, v in answer_to_idx.items()}
        if args.answer_idx in idx_to_answer:
            answer_text = idx_to_answer[args.answer_idx]

    # 可视化
    print(f"生成Grad-CAM可视化...")
    visualize_gradcam(
        image_path=args.image_path,
        model=grad_cam_model,
        target_layer=args.target_layer,
        class_idx=args.answer_idx,
        question=args.question,
        answer=answer_text,
        save_path=args.save_path
    )

# 在 testVisualVqa.py 中添加以下代码
def print_model_layers(model):
    print("可用的模型层名称:")
    for name, module in model.named_modules():
        print(name)

if __name__ == "__main__":
    main()
