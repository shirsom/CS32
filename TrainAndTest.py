#from __future__ import print_function, division
from config import BASE_LR, EPOCH_DECAY, DECAY_WEIGHT, NUM_EPOCHS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

class TrainAndTest(object):

  def __init__(self,
               model,
               datasets,
               model_path,
               config=None,
               lr=BASE_LR):
    self.model_ft = model
    self.dataset_manager = datasets
    self.accuracies = {}
    self.losses = {}
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.RMSprop(self.model_ft.parameters(), lr=BASE_LR)
    self.phases = ['Train', 'Val']
    self.model_path = model_path

  def train(self, num_epochs=NUM_EPOCHS):
    best_acc = 0

    for epoch in range(1, num_epochs + 1):
      print('-' * 10)
      print('Epoch {}/{}'.format(epoch, num_epochs))
      print('-' * 10)

      true_labels = []
      predicted_labels = []

      for phase in self.phases:
        if phase == 'Train':
          self.optimizer = self.lr_scheduler(epoch)
          self.model_ft.train()
        else:
          self.model_ft.eval()

        running_loss = 0.0
        running_corrects = 0

        for data in self.dataset_manager.dataloaders[phase]:
          inputs, labels = data
          inputs, labels = Variable(inputs), Variable(labels)

          self.optimizer.zero_grad()
          outputs = self.model_ft(inputs)
          _, preds = torch.max(outputs.data, 1)

          loss = self.criterion(outputs, labels)

          if phase == 'Train':
            loss.backward()
            self.optimizer.step()
          else:
            true_labels.extend(labels.numpy())
            predicted_labels.extend(preds.numpy())

          try:
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
          except:
            print('unexpected error, could not calculate loss or do a sum.')

        epoch_loss = running_loss / self.dataset_manager.dataset_sizes[phase]
        epoch_acc = running_corrects.item() / float(
            self.dataset_manager.dataset_sizes[phase])
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                   epoch_acc))

        if phase not in self.accuracies:
          self.accuracies[phase] = [epoch_acc]
        else:
          self.accuracies[phase].append(epoch_acc)

        if phase not in self.losses:
          self.losses[phase] = [epoch_loss]
        else:
          self.losses[phase].append(epoch_loss)

        if phase == 'Val':
          if epoch_acc > best_acc:
            best_acc = epoch_acc
            print('New best accuracy =', best_acc)
            torch.save(self.model_ft.state_dict(), self.model_path)


  def lr_scheduler(self, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    lr = init_lr * (DECAY_WEIGHT ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
      print('LR is set to {}'.format(lr))

    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr

    return self.optimizer
