import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Преобразование данных в тензоры PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Создание DataLoader для удобства обучения
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Определение архитектуры нейронной сети
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Определение слоев нейронной сети
        self.fc1 = nn.Linear(4, 64)  # Входной слой с 4 входными признаками и 64 нейронами
        self.fc2 = nn.Linear(64, 64)  # Скрытый полносвязный слой с 64 нейронами
        self.fc3 = nn.Linear(64, 3)   # Выходной слой с 3 нейронами для 3 классов в Iris dataset

    def forward(self, x):
        # Определение прямого прохода (forward pass)
        x = torch.relu(self.fc1(x))  # Применение функции активации ReLU к выходу первого слоя
        x = torch.relu(self.fc2(x))  # Применение функции активации ReLU к выходу второго слоя
        x = self.fc3(x)              # Выходной слой без функции активации (для CrossEntropyLoss)
        return x

# Создание экземпляра нейронной сети
model = NeuralNetwork()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()  # Функция потерь CrossEntropyLoss для задачи классификации
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Оптимизатор Adam с коэффициентом скорости обучения 0.001

# Обучение модели
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()     # Обнуление градиентов
        outputs = model(inputs)   # Прямой проход: получение предсказаний модели
        loss = criterion(outputs, targets)  # Вычисление функции потерь
        loss.backward()           # Обратное распространение: вычисление градиентов
        optimizer.step()          # Обновление параметров модели

# Предсказание меток классов на тестовом наборе
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)  # Выбор класса с наибольшей вероятностью
    y_pred = predicted.numpy()

# Оценка точности модели
precision = precision_score(y_test, y_pred, average='weighted')  # Вычисление точности модели
print("Precision of the neural network:", precision)

 #  этот код использует PyTorch для создания, обучения и оценки нейронной сети на наборе данных Iris.