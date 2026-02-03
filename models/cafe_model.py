"""
CAFE: Cross-modal Ambiguity Learning for Multimodal Fake News Detection
Implementation based on WWW 2022 paper

This implementation supports:
1. Full multimodal (text + image) detection
2. Text-only detection using transformer encoders
"""

import math
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from transformers import AutoModel


class FastCNN(nn.Module):
    """CNN-based text encoder for pre-extracted word embeddings"""
    
    def __init__(self, input_dim=200, channel=32, kernel_size=(1, 2, 4, 8)):
        super(FastCNN, self).__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, channel, kernel_size=kernel),
                    nn.BatchNorm1d(channel),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1)
                )
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_out = []
        for module in self.fast_cnn:
            out = module(x).squeeze(-1)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            x_out.append(out)
        x_out = torch.cat(x_out, 1)
        return x_out


class TransformerTextEncoder(nn.Module):
    """Transformer-based text encoder using pretrained models"""
    
    def __init__(self, model_name='xlm-roberta-base', output_dim=128, freeze_bert=False):
        super(TransformerTextEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        if freeze_bert:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.projection(pooled)


class EncodingPart(nn.Module):
    """Multimodal encoding module for text and image features"""
    
    def __init__(
        self,
        text_input_dim=200,
        image_input_dim=512,
        cnn_channel=32,
        cnn_kernel_size=(1, 2, 4, 8),
        shared_dim=128
    ):
        super(EncodingPart, self).__init__()
        
        # Text encoding
        self.shared_text_encoding = FastCNN(
            input_dim=text_input_dim,
            channel=cnn_channel,
            kernel_size=cnn_kernel_size
        )
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )
        
        # Image encoding
        self.shared_image = nn.Sequential(
            nn.Linear(image_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_encoding = self.shared_text_encoding(text)
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image)
        return text_shared, image_shared


class SimilarityModule(nn.Module):
    """Cross-modal similarity learning module"""
    
    def __init__(self, shared_dim=128, sim_dim=64):
        super(SimilarityModule, self).__init__()
        self.encoding = EncodingPart()
        
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(sim_dim * 2),
            nn.Linear(sim_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, text, image):
        text_encoding, image_encoding = self.encoding(text, image)
        text_aligned = self.text_aligner(text_encoding)
        image_aligned = self.image_aligner(image_encoding)
        sim_feature = torch.cat([text_aligned, image_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)
        return text_aligned, image_aligned, pred_similarity


class Encoder(nn.Module):
    """Variational encoder for ambiguity learning"""
    
    def __init__(self, input_dim=64, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, x):
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class AmbiguityLearning(nn.Module):
    """Cross-modal ambiguity learning using KL divergence"""
    
    def __init__(self, input_dim=64):
        super(AmbiguityLearning, self).__init__()
        self.encoder_text = Encoder(input_dim)
        self.encoder_image = Encoder(input_dim)

    def forward(self, text_encoding, image_encoding):
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1) / 2.
        skl = torch.sigmoid(skl)
        return skl


class UnimodalDetection(nn.Module):
    """Unimodal representation learning"""
    
    def __init__(self, shared_dim=128, prime_dim=16):
        super(UnimodalDetection, self).__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )

    def forward(self, text_encoding, image_encoding):
        text_prime = self.text_uni(text_encoding)
        image_prime = self.image_uni(image_encoding)
        return text_prime, image_prime


class CrossModule(nn.Module):
    """Cross-modal correlation learning"""
    
    def __init__(self, in_dim=64, out_dim=64):
        super(CrossModule, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_in = text.unsqueeze(2)
        image_in = image.unsqueeze(1)
        corre_dim = text.shape[1]
        similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze(-1)
        correlation_out = self.projection(correlation_p)
        return correlation_out


class CAFEModel(nn.Module):
    """
    CAFE: Cross-modal Ambiguity Learning for Multimodal Fake News Detection
    Full multimodal model for pre-extracted features
    """
    
    def __init__(self, feature_dim=96, h_dim=64):
        super(CAFEModel, self).__init__()
        self.encoding = EncodingPart()
        self.ambiguity_module = AmbiguityLearning()
        self.uni_repre = UnimodalDetection()
        self.cross_module = CrossModule()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2)
        )

    def forward(self, text_raw, image_raw, text_aligned, image_aligned):
        skl = self.ambiguity_module(text_aligned, image_aligned)
        text_enc, image_enc = self.encoding(text_raw, image_raw)
        text_prime, image_prime = self.uni_repre(text_enc, image_enc)
        correlation = self.cross_module(text_aligned, image_aligned)
        
        weight_uni = (1 - skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        
        text_final = weight_uni * text_prime
        img_final = weight_uni * image_prime
        corre_final = weight_corre * correlation
        
        final_feature = torch.cat([text_final, img_final, corre_final], 1)
        pre_label = self.classifier(final_feature)
        return pre_label


class TextOnlyClassifier(nn.Module):
    """
    Text-only fake news classifier using transformer encoder
    For Vietnamese dataset without images
    """
    
    def __init__(self, model_name='xlm-roberta-base', num_classes=2, hidden_dim=256, dropout=0.3):
        super(TextOnlyClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(pooled)
