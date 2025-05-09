import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Load the dataset
df = pd.read_csv("data/ecommerce_dummy.csv")

# Encode user_id and item_id as integers
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df['user'] = user_encoder.fit_transform(df['user_id'])
df['item'] = item_encoder.fit_transform(df['item_id'])

# Focus only on 'purchase' events for recommendations
df = df[df['event_type'] == 'purchase']

# Prepare data
X = df[['user', 'item']].values
y = torch.ones(len(X))  # All purchases = positive interactions

# Train-test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Define Matrix Factorization model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=10):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        return (user_emb * item_emb).sum(dim=1)

# Initialize the model
num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
embedding_dim = 10
model = MatrixFactorization(num_users, num_items, embedding_dim)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Convert training data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y[:len(X_train)], dtype=torch.float32)

# Train the model
epochs = 10
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor[:, 0], X_train_tensor[:, 1])
    loss = loss_fn(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Plot training loss
plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/recommender_model.pth")
print("✅ Recommender model saved at models/recommender_model.pth")

# Evaluate the model on test data
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.long)

with torch.no_grad():
    test_predictions = model(X_test_tensor[:, 0], X_test_tensor[:, 1])

test_loss = loss_fn(test_predictions, torch.ones(len(X_test_tensor), dtype=torch.float32))
print(f"\n🔍 Test Loss (MSE): {test_loss.item():.4f}")

# Recommend top items for a specific user
user_id = 0  # Change this to test for a different user
predicted_scores = []

with torch.no_grad():
    for item in range(num_items):
        score = model(torch.tensor([user_id]), torch.tensor([item]))
        predicted_scores.append((item, score.item()))

# Get top 5 recommendations
recommended_items = sorted(predicted_scores, key=lambda x: x[1], reverse=True)[:5]
recommended_item_ids = [item[0] for item in recommended_items]

# Display recommendations
print(f"\n🎯 Top 5 recommended items for user {user_id_encoder := user_encoder.inverse_transform([user_id])[0]}:")
for item in recommended_item_ids:
    print(f" - Item ID: {item_encoder.inverse_transform([item])[0]}")

# Save recommendations to CSV (optional for dashboard)
output_csv = "static/recommended_items.csv"
os.makedirs("static", exist_ok=True)
with open(output_csv, "w") as f:
    f.write("user_id,item_id,score\n")
    for item, score in recommended_items:
        f.write(f"{user_id_encoder},{item_encoder.inverse_transform([item])[0]},{round(score, 4)}\n")

print(f"\n📁 Recommendations saved to {output_csv}")
