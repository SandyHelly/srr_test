import torch
import pytorch_lightning as pl


class HARModule(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer):
        super().__init__()
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return {'loss': loss, 'batch_size': batch_size}
    
    def training_epoch_end(self, outputs):
        sum_loss = torch.stack([x['loss'] for x in outputs]).sum()
        num_samples = sum([x['batch_size'] for x in outputs])
        avg_loss = sum_loss / num_samples
        print('Train epoch Loss:', avg_loss.detach().numpy())
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return {'loss': loss, 'batch_size': batch_size}
    
    def validation_epoch_end(self, outputs):
        sum_loss = torch.stack([x['loss'] for x in outputs]).sum()
        num_samples = sum([x['batch_size'] for x in outputs])
        avg_loss = sum_loss / num_samples
        print('Val epoch Loss:', avg_loss.detach().numpy())
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return y_pred.detach(), y

    def configure_optimizers(self):
        return {'optimizer': self.optimizer}
