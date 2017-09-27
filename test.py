from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


plt.figure()
plt.axis("off")
plt.imshow(image)


image = image.reshape((image.shape[0] * image.shape[1], 3))

clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)
print clt

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)

# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
