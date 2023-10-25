import cv2
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from sklearn.cluster import DBSCAN
sns.set(style="whitegrid",rc={"figure.figsize": (19.2, 10.8)})

# Lib to data processing 
class dbscanAlgo:
    def __init__(self, img = '../data/face-mesh-ex.png', dataset = '../data/train_data.csv', x_df = 'screen_x', y_df = 'left_iris_y'):
        self.df                 = pd.read_csv(dataset) 
        self.x_df               = x_df
        self.y_df               = y_df
        self.X_train            = self.df[[self.x_df, self.y_df]]
        self.clustering         = DBSCAN(eps=12.5, min_samples=4).fit(self.X_train)
        self.DBSCAN_dataset     = self.X_train.copy()
        self.outliers           = ''
        self.img                = np.asarray(Image.open(img))
        self.path_save          = '../report/'
        self.path_              = '../data/'
        self.df_plot            = ''
        
    # Processing dataframe    
    def dfHandling(self):
        self.DBSCAN_dataset.loc[:,'Cluster'] = self.clustering.labels_ 
        self.DBSCAN_dataset.Cluster.value_counts().to_frame()
        self.outliers = self.DBSCAN_dataset[self.DBSCAN_dataset['Cluster']==-1]
        self.df = self.DBSCAN_dataset[self.DBSCAN_dataset['Cluster'] !=-1]

    # Creating clusters image to report
    def plotClusters(self):    
        fig = plt.figure()#figsize=(19.2, 10.8))
        sns.scatterplot(data = self.df, x = self.x_df, y = self.y_df, hue = self.df.Cluster, legend = "full", palette = "deep")
        plt.savefig(f'{self.path_save}clusters.png')        
        #plt.xlim(0, 1920)
        #plt.ylim(0, 1080)
        self.saveData('clusters', self.df)
        
        return fig
        
    # Saving new dataframe to future processing    
    def saveData(self, label, df):
        if type(label) == type(''):
            # Save df in csv file    
            self.df_plot = pd.DataFrame(df)
            self.df_plot.to_csv(f'{self.path_}/{label}.csv') 
   
    # Creating gaze points image to report    
    def plotGazePoints(self):
        fig = plt.figure() #figsize=(19.2, 10.8))
        plt.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'r', linestyle = '-')
        plt.xlabel = self.x_df
        plt.ylabel = self.y_df
        #plt.xlim(0, 1920)
        #plt.ylim(0, 1080)
        plt.savefig(f'{self.path_save}gaze_points.png')
        return fig
        
    # Creating heatmap overlay image to report    
    def plotDensity(self):
        fig = plt.figure()#figsize=(19.2, 10.8))
        plt.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'r', linestyle='-')

        sns.kdeplot(data=self.df, x=self.x_df, y=self.y_df, cmap="Reds", fill=True, alpha=.6)

        #plt.xlim(0, 1920)
        #plt.ylim(0, 1080)

        # Create a ScalarMappable object using a colormap
        cmap = plt.cm.get_cmap('Reds')  # Choose a colormap (e.g., 'Reds')
        sm = ScalarMappable(cmap=cmap)
        sm.set_array([])  # Set an empty array to enable color mapping

        # Add a colorbar using the ScalarMappable object
        cbar = plt.colorbar(sm)
        cbar.set_label('Density')

        plt.savefig(f'{self.path_save}heatmap.jpg', dpi=100)

        return fig
        
    # Processing data to overlay image     
    def overlayImageData(self):
        fig, ax = plt.subplots()
        ax.imshow(self.img, extent=[0, 1920, 0, 1080])

        sns.kdeplot(data=self.df, x=self.x_df, y=self.y_df, cmap="Reds", common_norm=False, levels=50, fill=True, alpha=.5)

        #plt.xlim(0, 1920)
        #plt.ylim(0, 1080)

        # Create a ScalarMappable object using a colormap
        cmap = plt.cm.get_cmap('Reds')  # Choose a colormap (e.g., 'Reds')
        sm = ScalarMappable(cmap=cmap)
        sm.set_array([])  # Set an empty array to enable color mapping

        # Add a colorbar using the ScalarMappable object
        cbar = plt.colorbar(sm)
        cbar.set_label('Density')

        fig.savefig(f'{self.path_save}overlay.jpg', dpi=100)
        return fig
        
    # Showing image to test       
    def showImage(self):
        cv2.imshow('Image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    def run(self):
        self.dfHandling()
        self.plotClusters()
        self.plotGazePoints()
        self.plotDensity()
        
    
        
if __name__ == '__main__':
    dbs = dbscanAlgo()
    dbs.showImage()
    input('Aperte enter para o processamento dos dados: ')
    dbs.run()
    print('------\nDados Processados\n------')