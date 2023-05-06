import os

import numpy as np
import rasterio as rio

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.exposure import equalize_adapthist

import matplotlib.pyplot as plt

class ClusteredBands:

    def __init__(self, rasters_list, scenename):
        self.rasters = rasters_list
        self.model_input = None
        self.width = 0
        self.height = 0
        self.depth = 0
        self.scene_name = scenename
        self.no_of_ranges = None
        self.models = None
        self.predicted_rasters = None
        self.s_scores = []
        self.inertia_scores = []
        self.bands_stack = []

    def set_raster_stack(self):
        band_list = []
        for image in self.rasters:
            with rio.open(image, 'r') as src:
                band = src.read(1)
                band = np.nan_to_num(band)
                band_list.append(band)
        bands_stack = np.dstack(band_list)

        # Prepare model input from bands stack
        self.width, self.height, self.depth = bands_stack.shape
        self.model_input = bands_stack.reshape(self.width * self.height, self.depth)

    def build_models(self, no_of_clusters_range):
        self.no_of_ranges = no_of_clusters_range
        models = []
        predicted = []
        inertia_vals = []
        s_scores = []
        y_pred = []
        for n_clust in no_of_clusters_range:
            quantized_raster = []
            kmeans = KMeans(n_clusters=n_clust)
            y_pred = kmeans.fit_predict(self.model_input)

            # Append model
            models.append(kmeans)

            # Calculate metrics
            s_scores.append(self._calc_s_score(y_pred))
            inertia_vals.append(kmeans.inertia_)

            # Append output image (classified)
            quantized_raster = np.reshape(y_pred, (self.width, self.height))
            plt.imsave(self.scene_name+'.png', quantized_raster,cmap='terrain')
            plt.rcParams.update(plt.rcParamsDefault)
            plt.close()
            predicted.append(quantized_raster)

        # Update class parameters
        self.models = models
        self.predicted_rasters = predicted
        self.s_scores = s_scores
        self.inertia_scores = inertia_vals

    def _calc_s_score(self, labels):
        s_score = silhouette_score(self.model_input, labels, sample_size=100)
        return s_score

    def show_clustered(self):
        for idx, no_of_clust in enumerate(self.no_of_ranges):
            title = 'Number of clusters: ' + str(no_of_clust)
            image = self.predicted_rasters[idx]
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.title(title)
            plt.imshow(image, cmap='Accent')
            plt.colorbar()
            plt.show()

    def show_inertia(self):
        plt.figure(figsize=(10, 10))
        plt.title('Inertia of the models')
        plt.plot(self.no_of_ranges, self.inertia_scores)
        plt.show()

    def show_silhouette_scores(self):
        plt.figure(figsize=(10, 10))
        plt.title('Silhouette scores')
        plt.plot(self.no_of_ranges, self.s_scores)
        plt.show()
