import torch
from modules import GNN
from dataset import load_data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train(data, model, optimizer, loss_fn, epochs=100, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0  # 记录验证损失没有改善的次数

    # 用于记录损失和准确率
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # 记录训练损失
        train_losses.append(loss.item())

        # 计算训练准确率
        train_predictions = out[data.train_mask].argmax(dim=1)
        train_accuracy = accuracy_score(data.y[data.train_mask].cpu(), train_predictions.cpu())
        train_accuracies.append(train_accuracy)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            out = model(data)  # 重新计算输出
            val_loss = loss_fn(out[data.val_mask], data.y[data.val_mask])
            val_predictions = out[data.val_mask].argmax(dim=1)
            val_accuracy = accuracy_score(data.y[data.val_mask].cpu(), val_predictions.cpu())

        # 记录验证损失和准确率
        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch}, Training Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy * 100:.2f}%, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

        # 检查是否需要更新最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0  # 重置计数器
        else:
            epochs_no_improve += 1

        # 检查是否需要早停
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # 训练结束后，加载最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, "model.pth")
    else:
        torch.save(model.state_dict(), "model.pth")

    # 绘制并保存损失曲线
    epochs_range = range(len(train_losses))

    plt.figure()
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig('loss_over_epochs.png')
    plt.close()

    # 绘制并保存准确率曲线
    plt.figure()
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig('accuracy_over_epochs.png')
    plt.close()

if __name__ == "__main__":
    # 加载数据
    npz_file_path = '/Users/zhangxiangxu/Downloads/3710_report/facebook.npz'
    data = load_data(npz_file_path)

    # 初始化模型、优化器和损失函数
    model = GNN(in_channels=data.num_features, out_channels=4) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 训练模型，添加 early stopping
    train(data, model, optimizer, loss_fn, epochs=100, patience=10)
    print(f"Number of training nodes: {data.train_mask.sum().item()}")
