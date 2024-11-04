import torch
import torch.nn as nn
import math
import numpy as np

class CustomLossM(nn.Module):
    def __init__(self, alpha=0.5, epsilon=1e-6):
        super(CustomLossM, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Directional accuracy with improved precision
        diff_true = y_true[:, 1:] - y_true[:, :-1]
        diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
        
        direction_true = torch.sign(diff_true)
        direction_pred = torch.sign(diff_pred)
        
        # Calculate the angle between true and predicted directions
        dot_product = (diff_true * diff_pred).sum(dim=1)
        norm_true = torch.norm(diff_true, dim=1)
        norm_pred = torch.norm(diff_pred, dim=1)
        
        # Use clamp to avoid NaN in acos
        cos_angle = torch.clamp(dot_product / (norm_true * norm_pred + self.epsilon), -1 + self.epsilon, 1 - self.epsilon)
        angle_error = torch.acos(cos_angle)
        
        # Normalize angle error to [0, 1] range
        direction_loss = angle_error / math.pi
        
        # Magnitude accuracy
        magnitude_loss = self.mse_loss(y_pred, y_true)
        
        # Combined loss
        loss = self.alpha * direction_loss.mean() + (1 - self.alpha) * magnitude_loss
        
        return loss
    

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        def _log_cosh(x):
            return x + torch.nn.functional.softplus(-2. * x) - torch.log(torch.tensor(2.))
        return torch.mean(_log_cosh(y_pred - y_true))
    

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5, epsilon=1e-6, direction_threshold=0.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.direction_threshold = direction_threshold
        self.mse_loss = nn.MSELoss()
        
    def calculate_direction_accuracy(self, direction_true, direction_pred):
        """Calculate direction accuracy similar to sklearn's accuracy_score"""
        correct_directions = (direction_true == direction_pred).float()
        return correct_directions.mean()
    
    def calculate_direction_loss(self, y_true, y_pred):
        """Calculate direction loss considering day-to-day movements"""
        # Calculate day-to-day differences
        diff_true = y_true[:, 1:] - y_true[:, :-1]
        diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
        
        # Get directions with threshold to handle small changes
        direction_true = torch.sign(diff_true)
        direction_pred = torch.sign(diff_pred)
        
        # Calculate binary direction loss (similar to accuracy)
        binary_direction_loss = 1.0 - self.calculate_direction_accuracy(direction_true, direction_pred)
        
        # Calculate magnitude-aware direction loss
        # Normalize the differences
        norm_true = torch.norm(diff_true, dim=1, keepdim=True)
        norm_pred = torch.norm(diff_pred, dim=1, keepdim=True)
        
        normalized_true = diff_true / (norm_true + self.epsilon)
        normalized_pred = diff_pred / (norm_pred + self.epsilon)
        
        # Calculate cosine similarity
        cosine_sim = (normalized_true * normalized_pred).sum(dim=1)
        cosine_sim = torch.clamp(cosine_sim, -1 + self.epsilon, 1 - self.epsilon)
        
        # Convert to angle error (0 to π)
        angle_error = torch.acos(cosine_sim)
        
        # Normalize to [0, 1] range
        normalized_angle_error = angle_error / math.pi
        
        # Combine binary and magnitude-aware direction losses
        direction_loss = 0.5 * binary_direction_loss + 0.5 * normalized_angle_error.mean()
        
        return direction_loss

    def calculate_trend_loss(self, y_true, y_pred):
        """Calculate loss for overall trend prediction"""
        trend_true = y_true[:, -1] - y_true[:, 0]
        trend_pred = y_pred[:, -1] - y_pred[:, 0]
        
        trend_direction_true = torch.sign(trend_true)
        trend_direction_pred = torch.sign(trend_pred)
        
        trend_accuracy = (trend_direction_true == trend_direction_pred).float().mean()
        trend_loss = 1.0 - trend_accuracy
        
        return trend_loss

    def forward(self, y_pred, y_true):
        # Calculate main direction loss (day-to-day movements)
        direction_loss = self.calculate_direction_loss(y_true, y_pred)
        
        # Calculate trend loss (overall sequence direction)
        trend_loss = self.calculate_trend_loss(y_true, y_pred)
        
        # Calculate magnitude loss (MSE)
        magnitude_loss = self.mse_loss(y_pred, y_true)
        
        # Combine all losses with weights
        direction_weight = self.alpha
        trend_weight = self.alpha * 0.2  # Give some weight to overall trend
        magnitude_weight = 1.0 - self.alpha
        
        total_loss = (direction_weight * direction_loss + 
                     trend_weight * trend_loss + 
                     magnitude_weight * magnitude_loss)
        
        return total_loss

class WeightedDirectionLoss(nn.Module):
    """Alternative loss function that weights direction errors by their magnitude"""
    def __init__(self, alpha=0.5, epsilon=1e-6):
        super(WeightedDirectionLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        # Calculate day-to-day differences
        diff_true = y_true[:, 1:] - y_true[:, :-1]
        diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
        
        # Calculate magnitude of changes
        magnitude_true = torch.abs(diff_true)
        
        # Calculate direction indicators
        direction_true = torch.sign(diff_true)
        direction_pred = torch.sign(diff_pred)
        
        # Calculate weighted direction errors
        direction_errors = (direction_true != direction_pred).float()
        weighted_direction_loss = (direction_errors * magnitude_true).mean()
        
        # Calculate magnitude loss
        magnitude_loss = self.mse_loss(y_pred, y_true)
        
        # Combine losses
        total_loss = self.alpha * weighted_direction_loss + (1 - self.alpha) * magnitude_loss
        
        return total_loss