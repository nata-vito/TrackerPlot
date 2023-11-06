import os
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from sklearn import metrics
from sklearn.cluster import DBSCAN
sns.set(style="whitegrid",rc={"figure.figsize": (19.2, 10.8)})

# Lib to data processing 
class TrackerPlot:
    def __init__(self, img = '../data/grocery.jpg', dataset = '../data/train_data.csv', x_df = 'screen_x', y_df = 'left_iris_y', test = 'calibration'):
        self.df                 = pd.read_csv(dataset) 
        self.x_df               = x_df
        self.y_df               = y_df
        self.points             = self.df[[self.x_df, self.y_df]]
        self.clustering         = DBSCAN(eps=16, min_samples=4).fit(self.points)
        self.labels             = self.clustering.labels_
        self.DBSCAN_dataset     = self.points.copy()
        self.outliers           = ''
        self.test_type          = test

        # Number of clusters in labels, ignoring noise if present.
        self.n_clusters_        = len(set(self.labels)) - (1 if -1 in self.labels else 0) 
        self.n_noise_           = list(self.labels).count(-1)

        self.img                = np.asarray(Image.open(img))
        self.path_save          = '../report/'
        self.path_              = '../data/'
        self.df_plot            = ''
        self.plot_info          = []
        self.report_cluster     = {}
    
    def plotNoProcessedPoints(self):
        # Plot the data
        fig = plt.figure(figsize=(19.2, 10.8))
        sns.scatterplot(data = self.df, x = self.x_df, y = self.y_df, legend = "full", palette = "deep")
         
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        self.saveData(type_test=self.test_type, label = f'no_processed_{self.x_df}_{self.y_df}', df = self.df)
        plt.savefig(f'{self.path_save}/{self.test_type}/plot/no_processed_{self.x_df}_{self.y_df}.png')
        
    
    def clusterMetrics(self):
        if self.n_noise_ > 0:
            self.report_cluster = {
                'params': (self.x_df, self.y_df),
                'n_cluster': self.n_clusters_,
                'noise_points': self.n_noise_,
                'silhouette_coef': metrics.silhouette_score(self.points, self.labels) 
            }
        else:
            self.report_cluster = {
                'params': (0, 0),
                'n_cluster': 0,
                'noise_points': 0,
                'silhouette_coef': 0 
            }
            
           
        print('----' * 10)
        print(self.report_cluster['params'])
        print("Estimated number of clusters: %d" % self.report_cluster['n_cluster'])
        print("Estimated number of noise points: %d" % self.report_cluster['noise_points'])
        print("Silhouette Coefficient: %.3f" % self.report_cluster['silhouette_coef'])
        print('----' * 10)



    # Processing dataframe after clustering    
    def dfHandling(self):
        self.DBSCAN_dataset.loc[:,'Cluster'] = self.clustering.labels_ 
        self.DBSCAN_dataset.Cluster.value_counts().to_frame()
        self.outliers = self.DBSCAN_dataset[self.DBSCAN_dataset['Cluster']==-1]
        self.df = self.DBSCAN_dataset[self.DBSCAN_dataset['Cluster'] !=-1]

    # Creating clusters image to report
    def plotClusters(self):    
        fig = plt.figure(figsize=(19.2, 10.8))
        sns.scatterplot(data = self.df, x = self.x_df, y = self.y_df, hue = self.df.Cluster, legend = "full", palette = "deep")      
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        self.saveData(type_test = self.test_type, label = f'clusters_{self.x_df}_{self.y_df}', df = self.df)
        plt.savefig(f'{self.path_save}/{self.test_type}/plot/clusters_{self.x_df}_{self.y_df}.png')
        
        return fig
        
    # Saving new dataframe to future processing and make dirs to report 
    def saveData(self, type_test, label, df):
        if type(label) == type(''):
            path_data = f'{self.path_save}/{type_test}/data/'
            path_plot = f'{self.path_save}/{type_test}/plot/'
            try:
                os.makedirs(path_data, exist_ok=True)
                os.makedirs(path_plot, exist_ok=True)
            except OSError as error:
                print("Directory '%s' can not be created" % path_data) 
            
            # Save df in csv file    
            self.df_plot = pd.DataFrame(df)
            self.df_plot.to_csv(f'{self.path_save}/{type_test}/data/{label}.csv') 
   
    # Creating gaze points image to report    
    def plotGazePoints(self):
        fig = plt.figure(figsize=(19.2, 10.8))
        plt.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'r', linestyle = '-')
        plt.xlabel = self.x_df
        plt.ylabel = self.y_df
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        plt.savefig(f'{self.path_save}/{self.test_type}/plot/gaze_points_{self.x_df}_{self.y_df}.png')
        return fig
        
    # Creating heatmap overlay image to report    
    def plotDensity(self):
        fig = plt.figure()#figsize=(19.2, 10.8))
        plt.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'r', linestyle='-')

        sns.kdeplot(data=self.df, x=self.x_df, y=self.y_df, cmap="Reds", fill=True, alpha=.6)

        plt.xlim(0, 1920)
        plt.ylim(0, 1080)

        # Create a ScalarMappable object using a colormap
        cmap = plt.cm.get_cmap('Reds')  # Choose a colormap (e.g., 'Reds')
        sm = ScalarMappable(cmap=cmap)
        sm.set_array([])  # Set an empty array to enable color mapping

        # Add a colorbar using the ScalarMappable object
        cbar = plt.colorbar(sm)
        cbar.set_label('Density')

        plt.savefig(f'{self.path_save}/{self.test_type}/plot/heatmap_{self.x_df}_{self.y_df}.jpg', dpi=100)

        return fig
        
    # Processing data to overlay image     
    def overlayImageData(self):
        fig, ax = plt.subplots()
        ax.imshow(self.img, extent=[0, 1920, 0, 1080])

        sns.kdeplot(data=self.df, x=self.x_df, y=self.y_df, cmap="Reds", common_norm=False, levels=50, fill=True, alpha=.5)

        plt.xlim(0, 1920)
        plt.ylim(0, 1080)

        # Create a ScalarMappable object using a colormap
        cmap = plt.cm.get_cmap('Reds')  # Choose a colormap (e.g., 'Reds')
        sm = ScalarMappable(cmap=cmap)
        sm.set_array([])  # Set an empty array to enable color mapping

        # Add a colorbar using the ScalarMappable object
        cbar = plt.colorbar(sm)
        cbar.set_label('Density')

        fig.savefig(f'{self.path_save}/{self.test_type}/plot/overlay_{self.x_df}_{self.y_df}.jpg', dpi=100)
        return fig

    # Showing image to test       
    def showImage(self):
        cv2.imshow('Image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        self.plotNoProcessedPoints()
        self.dfHandling()
        self.plotClusters()
        self.clusterMetrics()
        self.plotGazePoints()
        self.plotDensity()
        self.overlayImageData()
        
        
        return self.report_cluster
        
    def typeAll(self):
        print(type(self.df))
        print(type(self.x_df))
        print(type(self.y_df))
        print(type(self.points))
        print(type(self.clustering))
        print(type(self.labels))
        print(type(self.DBSCAN_dataset))
        print(type(self.outliers))
        print(type(self.test_type))
        print(type(self.n_clusters_))
        print(type(self.n_noise_))
        print(type(self.img))
        print(type(self.path_save))
        print(type(self.path_))
        print(type(self.plot_info))
        print(type(self.report_cluster))