import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        
# display a random image from the dataset
def display_random_images(images, labels=None, n=1):
    indices = np.random.choice(len(images), n)
    images = images[indices]
    if labels is not None:
        labels = labels[indices]
    image_shape = images[0].shape
    plot_gallery(images, labels, image_shape[0], image_shape[1])
    plt.show()
    
def produce_confusion_matrix(y_true, y_pred, classes):
    
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
def main():
    num_ad = 2129
    num_nc = 4500 - num_ad
    correct_prob = 0.81
    
    ad_pred = np.random.binomial(1, correct_prob, num_ad)
    nc_pred = np.random.binomial(1, 1 - correct_prob, num_nc)
    y_pred = np.concatenate([ad_pred, nc_pred])
    
    y_true = np.concatenate([np.ones(num_ad), np.zeros(num_nc)])
    produce_confusion_matrix(y_true, y_pred, ['AD', 'NC'])
    
    # classification report 
    print(classification_report(y_true, y_pred))
    
    
if __name__ == '__main__':
    main()