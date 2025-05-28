import pandas as pd

from ._task import Task

class TaskName(Task):

    def __init__(self, args):

        super().__init__(args)      # initialize dataset name, data_df
        self.task_type = None

    
    def get_data_df(self):
        """
        Return data as a df. Limited to num_samples examples.
        
        DF columns:
        - input_args:   each sample (and its components) stored as a tuple
        - targets:      target outputs / gold answers
        - ...           any other metadata columns
        """
        
        # Create data_df
        data_df = ...

        self.set_data_df(data_df)

        self.get_num_samples()
        
        return self.data_df


    def score(self, predictions):
        """
        Compute task performance.
        """

        metrics, errors = ...

        
        return metrics, errors


