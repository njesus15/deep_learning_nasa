from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_predictions(filename):
    with open(filename) as f:
        data = f.readlines()
        n = len(data)
        print(n)
        predictions = []
        for line in data[0:-1: 8]: # get 10 percent of data
            prediction = line.split("\t")
            label = int(prediction[1])
            probs_str = prediction[2][1:-2].split(",")
            probs = [float(x) for x in probs_str]
            sum_probs = sum(probs)
            probs = [x / sum_probs for x in probs]
            predictions.append([prediction[0], label, probs, probs.index(max(probs))])
        return predictions


def results_2d(predictions):
    hidden_features = [np.array(x[2]) for x in predictions]
    #pca = PCA(n_components=20)
    #pca_result = pca.fit_transform(hidden_features)
    #print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))
    ##Variance PCA: 0.993621154832802

    #Run T-SNE on the PCA features.
    tsne = TSNE(n_components=2, verbose = 1)
    tsne_results = tsne.fit_transform(hidden_features)
    return tsne_results

def visualize(tsne_results, predictions, label_names):
    labels = [x[1] for x in predictions]
    y_test = np.array(labels)
    y_test_cat = np_utils.to_categorical(y_test[:], num_classes = len(label_names))
    color_map = np.argmax(y_test_cat, axis=1)
    plt.figure(figsize=(8,8))
    plt.title('Dataset 1 & 2 w/ Rep Layer of NN model Exc 001')
    markers = ["o", ".", "1", "x", '*', '^']
#    img = Image.open('dataset/combined/left/00000002_5.jpg')  # opens the file using Pillow - it's not an array yet
#    img.thumbnail((64, 64), Image.ANTIALIAS)  # resizes image in-place

    #for i in range(0, len(predictions), 10):
        #img = Image.open(predictions[i][0])  # opens the file using Pillow - it's not an array yet
        #img.thumbnail((64, 64), Image.ANTIALIAS)  # resizes image in-place
        #imgplot = plt.imshow(img,
        #    extent=[tsne_results[i, 0],tsne_results[i, 0] + 2.5,
        #    tsne_results[i, 1],tsne_results[i, 1] + 2.5], zorder=1)

    for cl in range(len(label_names)):
        print(cl)
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(tsne_results[indices,0], tsne_results[indices, 1],
            label=label_names[cl], marker=markers[cl])



    # plt.xlim([-100,100])
    # plt.ylim([-100,100])

    plt.legend()
    plt.savefig('PredImages.pdf', format='pdf', dpi=900)
    plt.show()


def find_wrong(predictions):
    wrong = []
    for pred in predictions:
        if pred[1] != pred[3]:
            wrong_pred = pred[:]
            wrong_pred[1] = 4 #label misprediction
            wrong.append(wrong_pred)
    return wrong

def relabel_wrong(features, predictions):
    for i in range(len(features)):
        if predictions[i][1] != predictions[i][3]:
            wrong_pred = features[i][:]
            wrong_pred[1] = 3
            features.append(wrong_pred)

# img = Image.open('dataset/combined/left/00000002_5.jpg')  # opens the file using Pillow - it's not an array yet
# img.thumbnail((64, 64), Image.ANTIALIAS)  # resizes image in-place
# fig = plt.figure(figsize=(5,5))
#
# imgplot = plt.imshow(img, extent=[0,.25/2,0,.25/2])
# imgplot = plt.imshow(img, extent=[.5,.75,.5,.75])
# plt.scatter(np.random.rand(10), np.random.rand(10))
# # imgplot = plt.imshow(img, extent=[4,4.5,4,4.5])
# plt.show()

# features = load_predictions("vgg16_features_rand.txt")
features = load_predictions('/Users/jesusnavarro/Desktop/trail_project/Pickledata/visualize_001_002_nn_exc_001.txt')
# features = features[::10]

# predictions = load_predictions("deep_predictions.txt")
# relabel_wrong(features, predictions)

# features.extend(predictions)

output_2d = results_2d(features)
print("output_2d=", output_2d[0])

visualize(output_2d, features, ["1-left", "1-center", "1-right", '2-left', '2-center', '2-right'])
