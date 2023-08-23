import mlflow
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

class NeuralNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def loss(self,logit,labels):
        loss = nn.CrossEntropyLoss()
        loss = loss(logit,labels)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch,batch_idx):
        x , y = train_batch
        logit = self.forward(x)
        loss = self.loss(logit,y)
        self.log('train loss',loss)
        return loss
    
    def test_step(self,test_batch,batch_idx):
        x,y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)
        
class dataModule(pl.LightningDataModule):
    def setup(self, stage: str) -> None:
        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
            )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
            )
        
        self.train = training_data
        self.test = test_data
    
    def train_dataloader(self):
        train_dataloader = DataLoader(self.train, batch_size=64)
        return train_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(self.test, batch_size=64)
        return test_dataloader
    
data = dataModule()
model = NeuralNetwork()
trainer = pl.Trainer(max_epochs=10,enable_progress_bar=True,accelerator="auto")

mlflow.pytorch.autolog(registered_model_name='pytorch-lightning-sample-model')
with mlflow.start_run() as run:
    trainer.fit(model,data)


