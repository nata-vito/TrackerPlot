from tracker_plot import TrackerPlot

class ProcessingPipelie():
    def __init__(self) -> None:
        self.best_result = {
            'sil_coef': 0,
            'params': ''
        }
        self.last_result = []

    def PlotDfRelationCollunms(self, dataset = '../data/train_data.csv', test_type = 'test'):
        
        plot_iris_left_x_y = TrackerPlot(dataset = dataset, x_df='left_iris_x', y_df='left_iris_y', test = test_type)
        plot_iris_right_x_y = TrackerPlot(dataset = dataset, x_df='right_iris_x', y_df='right_iris_y', test = test_type)
        plot_iris_left_x_right_y = TrackerPlot(dataset = dataset, x_df='left_iris_x', y_df='right_iris_y', test = test_type)
        plot_iris_right_x_left_y = TrackerPlot(dataset = dataset, x_df='right_iris_x', y_df='left_iris_y', test = test_type)
        
        if test_type == 'calibration':
            plot_iris_left_x_screen_y = TrackerPlot(dataset = dataset, x_df='left_iris_x', y_df='screen_y', test = test_type)
            plot_iris_screen_x_left_y = TrackerPlot(dataset = dataset, x_df='screen_x', y_df='left_iris_y', test = test_type)
            plot_iris_right_x_screen_y = TrackerPlot(dataset = dataset, x_df='right_iris_x', y_df='screen_y', test = test_type)
            plot_iris_screen_x_right_y = TrackerPlot(dataset = dataset, x_df='screen_x', y_df='right_iris_y', test = test_type)    
            plot_screen_x_screen_y = TrackerPlot(dataset = dataset, x_df='screen_x', y_df='screen_y', test = test_type) 
            #self.last_result.append(plot_screen_x_screen_y.run())
            self.last_result.append(plot_iris_left_x_screen_y.run())
            self.last_result.append(plot_iris_screen_x_left_y.run())
            self.last_result.append(plot_iris_right_x_screen_y.run())
            self.last_result.append(plot_iris_screen_x_right_y.run())
    

        self.last_result.append(plot_iris_left_x_y.run())
        self.last_result.append(plot_iris_right_x_y.run())
        self.last_result.append(plot_iris_left_x_right_y.run())
        self.last_result.append(plot_iris_right_x_left_y.run())
        
    
    
if __name__ == '__main__':
    pro = ProcessingPipelie()
    print('-' * 20)
    print('calibration')
    pro.PlotDfRelationCollunms(dataset = '../data/train_data.csv', test_type = 'calibration')
    print('-' * 20)
    print('session')
    pro.PlotDfRelationCollunms(dataset = '../data/session_data.csv', test_type = 'session')
    print(pro.last_result)