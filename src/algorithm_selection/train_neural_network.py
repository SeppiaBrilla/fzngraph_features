import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random

class Model(nn.Module):
    def __init__(self, in_shape:int, out_shape:int) -> None:
        super().__init__()

        layer_size = 300
        self.in_linear = nn.Linear(in_shape, layer_size * 2)
        self.first_hidden = nn.Linear(layer_size * 2, layer_size * 4)
        self.second_hidden = nn.Linear(layer_size * 4, layer_size * 4)
        self.third_hidden = nn.Linear(layer_size * 4, layer_size * 2)
        self.last_hidden = nn.Linear(layer_size * 2, layer_size)
        self.out_layer = nn.Linear(layer_size, out_shape)

        self.activation = F.leaky_relu
        self.out_activation = F.softmax

    def forward(self, X:torch.Tensor) -> torch.Tensor:

        out = self.in_linear(X)
        out = self.activation(out)
        out = self.first_hidden(out)
        out = self.activation(out)
        out = self.second_hidden(out)
        out = self.activation(out)
        out = self.third_hidden(out)
        out = self.activation(out)
        out = self.last_hidden(out)
        out = self.activation(out)
        out = self.out_layer(out)
        out = self.out_activation(out, dim=1)

        return out

@torch.no_grad
def validate_model(model:Model, dataset:DataLoader, loss) -> float:
    running_loss = 0.

    model.eval()
    for _, (x, y) in enumerate(dataset):
        out = model(x)
        running_loss += float(loss(out, y).item())


    return running_loss/len(dataset)

def train_nn(X_train:torch.Tensor, y_train:torch.Tensor, epochs:int=300) -> Model:
    model = Model(X_train.shape[1], y_train.shape[1])

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, shuffle=True)
    n = int(len(y_train) * .8)
    X_val = X_train[n:]
    y_val = y_train[n:]

    X_train = X_train[:n]
    y_train = y_train[:n]

    train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
    validation_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)


    best_model = Model(X_train.shape[1], y_train.shape[1])
    best_loss = torch.tensor(float('inf'))
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            outputs = model(x)

            loss = F.cross_entropy(outputs, y)
            loss.backward()

            optimizer.step()

        scheduler.step()
        train_loss = validate_model(model, train_dataloader, F.cross_entropy)
        val_loss = validate_model(model, validation_dataloader, F.cross_entropy)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model.load_state_dict(model.state_dict())

        print(f'\repoch: {epoch}, train loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, best_loss: {best_loss:.3f}', end='\r')

    print()
    return model

@torch.no_grad
def test_nn(model:Model, X_test:torch.Tensor, y_test:torch.Tensor, test_data:list[dict]) -> dict:
    pred = model(X_test).argmax(dim=1)
    accuracy = accuracy_score(y_test.argmax(dim=1), pred)

    pred_time = 0
    chuffed_time = 0
    cp_sat_time_time = 0
    vbs_time = 0
    for i, e in enumerate(test_data):
        x = X_test[i].reshape((1, X_test.shape[1]))
        pred = model(x)[0].argmax(dim=0)
        if pred == 0:
            pred_time += e['chuffed']
        elif pred == 1 or pred == 2:
            pred_time += e['cp-sat']
        else:
            raise Exception(pred)
        chuffed_time += e['chuffed']
        cp_sat_time_time += e['cp-sat']
        vbs_time += min(e['chuffed'], e['cp-sat'])

    print(f"accuracy: {accuracy:.3f}")
    print(f"predicted time as a percentage of the virtual best: {pred_time/vbs_time:.3f}")
    print(f"cuffed time as a percentage of the virtual best: {chuffed_time/vbs_time:.3f}")
    print(f"cp-sat time as a percentage of the virtual best: {cp_sat_time_time/vbs_time:.3f}")
    print(f"predicted time as a percentage of the chuffed time: {pred_time/chuffed_time:.3f}")
    print(f"predicted time as a percentage of the cp-sat time: {pred_time/cp_sat_time_time:.3f}")

    return {
        'accuracy': float(accuracy),
        'clf_time': float(pred_time),
        'vbs_time': float(vbs_time),
        'chuffed_time': float(chuffed_time),
        'cp-sat_time': float(cp_sat_time_time),
        'clf_vbs': float(pred_time/vbs_time),
        'chuffed_vbs': float(chuffed_time/vbs_time),
        'cp-sat_vbs': float(cp_sat_time_time/vbs_time),
        'clf_chuffed': float(pred_time/chuffed_time),
        'clf_cp-sat': float(pred_time/cp_sat_time_time)
        }

def train_and_test_nn(train_data:list[dict], test_data:list[dict], reduce:bool) -> dict:
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    X_train = np.array([e['features'] for e in train_data])

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_train = torch.tensor(X_train, dtype=torch.float32)

    y_label = {0: [1., 0., 0.], 1: [0., 1., 0.], 2: [0., 0., 1.]}
    y_train = torch.tensor([y_label[e['label']] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    X_test = scaler.transform(X_test)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor([y_label[e['label']] for e in test_data])

    model = train_nn(X_train, y_train)

    return test_nn(model, X_test, y_test, test_data)
