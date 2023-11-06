from tracker_plot import TrackerPlot

if __name__ == '__main__':
    plot = TrackerPlot(test='gaze_point_prot', dataset='../data/gaze_points.csv', x_df='x', y_df='y')
    plot.run()
    plot.typeAll()