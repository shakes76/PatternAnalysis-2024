import dataset.py
import train.py
import modules.py



image = dataset.load_data_3D('semantic_MRs_anon', normImage=False, categorical=False, dtype=np.float32, getAffines=False, orient=False, early_stop=False)
label = dataset.load_data_3D('semantic_labels_anon', normImage=False, categorical=False, dtype=np.uint8, getAffines=False, orient=False, early_stop=False)

image = torch.tensor(image, dtype=torch.float32)
label = torch.tensor(label, dtype=torch.long)

train_loader = DataLoader(image, batch_size=1, shuffle=True)
train.train_model(model, trainloader, criterion, optimizer, num_epochs=25)
