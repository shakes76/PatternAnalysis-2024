import torch
from sklearn.metrics import accuracy_score, classification_report

def predict(model, graph_data):
    model.eval()
    with torch.no_grad():
        out, _ = model(graph_data)
        pred = out.argmax(dim=1).cpu()

    accuracy = accuracy_score(graph_data.y.cpu(), pred)
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(graph_data.y.cpu(), pred))
