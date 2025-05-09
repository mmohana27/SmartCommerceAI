import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import csv

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Simulate daily sales for a product (100 days)
days = 100
t = np.arange(0, days)
sales = 50 + 10 * np.sin(0.2 * t) + np.random.normal(0, 3, days)

# Prepare data for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(sales, seq_length)

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # shape: [samples, seq_len, 1]
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define LSTM model
class DemandLSTM(nn.Module):
    def __init__(self):
        super(DemandLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take the output of last timestep
        return self.fc(out)

model = DemandLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(100):
    model.train()
    output = model(X_tensor)
    loss = criterion(output.squeeze(), y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

# Predict future demand (next 10 days)
model.eval()
future_preds = []
input_seq = X_tensor[-1].unsqueeze(0)  # shape: [1, seq_len, 1]

with torch.no_grad():
    for _ in range(10):
        pred = model(input_seq)  # shape: [1, 1]
        future_preds.append(pred.item())
        pred_formatted = pred.unsqueeze(2)  # [1, 1, 1] for concatenation
        input_seq = torch.cat((input_seq[:, 1:, :], pred_formatted), dim=1)

# Print predictions
print("Future demand predictions for next 10 days:")
print([round(p, 2) for p in future_preds])

# Plot actual + forecast
plt.figure(figsize=(10, 4))
plt.plot(np.arange(0, len(sales)), sales, label="Historical Sales")
plt.plot(np.arange(len(sales), len(sales)+10), future_preds, label="Forecast", color='red')
plt.xlabel("Day")
plt.ylabel("Demand")
plt.title("Demand Forecast for Next 10 Days")
plt.legend()
plt.tight_layout()
plt.show()

# Save predictions to CSV (optional for dashboard)
forecast_csv = os.path.join(os.path.dirname(__file__), '../static/demand_forecast.csv')
os.makedirs(os.path.dirname(forecast_csv), exist_ok=True)
with open(forecast_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time Step', 'Forecast'])
    for i, val in enumerate(future_preds):
        writer.writerow([i, round(val, 2)])
