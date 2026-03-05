import torch
import torch.nn as nn
from model import GraphHead

class EnsembleModel(nn.Module):
    """
    Ensemble model for net parasitic capacitance prediction based on Algorithm 2.
    Uses multiple models with thresholds to determine final prediction.
    """
    def __init__(self, args, device, thresholds):
        """
        Initialize ensemble model with multiple GraphHead models.
        
        Args:
            args: Arguments containing model configuration
            device: Device to run models on
            thresholds: List of max prediction values for each model
        """
        super().__init__()
        self.thresholds = thresholds
        self.num_models = len(thresholds) + 1  # K models
        self.device = device
        self.task = args.task
        
        # Create K models
        self.models = nn.ModuleList([
            GraphHead(
                args.hid_dim, 1, num_layers=args.num_gnn_layers, 
                num_head_layers=args.num_head_layers, 
                use_bn=args.use_bn, drop_out=args.dropout, activation=args.act_fn, 
                src_dst_agg=args.src_dst_agg, max_dist=args.max_dist,
                task=args.task
            ).to(device) for _ in range(self.num_models)
        ])
    
    def forward(self, batch):
        """
        Forward pass implementing Algorithm 2 logic.
        During training, train all models with all data.
        During evaluation, follow Algorithm 2 to determine final predictions.
        """
        if self.training:
            # During training, we train all models with all data
            all_preds = []
            for i, model in enumerate(self.models):
                pred, true = model(batch)
                all_preds.append(pred)
            
            # Return predictions from the first model during training
            # (all models will be trained using their respective losses)
            return all_preds[0], batch.edge_label
        else:
            # During evaluation, apply Algorithm 2
            batch_size = batch.edge_label.size(0)
            final_pred = torch.zeros(batch_size, 1, device=self.device)
            
            # Step 1: initial prediction with M1
            p_n, _ = self.models[0](batch)
            final_pred = p_n
            
            # Steps 2-5: iterate through remaining models
            for i in range(1, self.num_models):
                p_prime, _ = self.models[i](batch)
                
                # Apply threshold logic from Algorithm 2
                # If p'(n) > max_i-1, then p(n) = p'(n)
                mask = (p_prime > self.thresholds[i-1]).squeeze()
                final_pred[mask] = p_prime[mask]
            
            return final_pred, batch.edge_label

    def train_step(self, batch, optimizers):
        """
        Custom training step that trains all models in the ensemble
        
        Args:
            batch: Input batch data
            optimizers: List of optimizers for each model
        
        Returns:
            Average loss, predictions, and ground truth
        """
        total_loss = 0
        
        # Train each model separately
        for i, model in enumerate(self.models):
            optimizers[i].zero_grad()
            
            # Forward pass through current model
            pred, y = model(batch)
            
            # 边级任务: y 格式为 [regression_label, classification_label]
            if self.task == 'regression':
                true = y[:, 0].view(-1, 1) if y.dim() > 1 and y.size(1) == 2 else y.view(-1, 1)
                loss = torch.nn.MSELoss()(pred, true)
            else:
                true = y[:, 1].long() if y.dim() > 1 and y.size(1) == 2 else y.long()
                true = true.view(-1, 1)
                loss = torch.nn.BCEWithLogitsLoss()(pred, true.float())
            
            # Backward pass
            loss.backward()
            optimizers[i].step()
            
            total_loss += loss.item()
        
        # Return average loss and predictions from first model
        pred, y = self.models[0](batch)
        
        # 提取正确的标签维度
        if self.task == 'regression':
            true = y[:, 0].view(-1, 1) if y.dim() > 1 and y.size(1) == 2 else y.view(-1, 1)
        else:
            true = y[:, 1].long() if y.dim() > 1 and y.size(1) == 2 else y.long()
        
        avg_loss = total_loss / self.num_models
        
        return avg_loss, pred.detach(), true.detach()