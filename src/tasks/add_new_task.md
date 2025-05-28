## How to Add a New Task/Dataset

### New Task Type

If the task type of the dataset is not yet existent, create a new task type file in the `tasks` directory. Copy the contents of `_template.py` for starter code. Implement the abstract methods, such as `get_data_df()` which is responsible for loading, setting, and returning a pd.DataFrame of the test samples, and `score()` which implements the appropriate scoring function for computing task performance on the dataset.

### Dataset Creation

Within the appropriate task type file, add to `get_data_df()` to implement the loading of the dataset into a pd.DataFrame object.

### Other Utilities

Add the dataset name to `src/tasks/__init__.py` so the correct Task subclass/task type is used for the dataset. Also indicate whether, during uncertainty scoring, we should use the original question versus parsed atomic questions for sampling $K$ assertions for consistency-based measurement of models' intrinsic confidence.

_If a new task type was created_, add the corresopnding task prompts to `src/prompts/task_prompts.py` and, if applicable, `src/prompts/input_prompts.py`, and create relevant entries in `src/prompts/__init__.py`.