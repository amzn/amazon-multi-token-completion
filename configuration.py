DATA_PATH = None
HOME_DIR = '.'

SAGEMAKER_ROLE = None
SAGEMAKER_BUCKET = None

if DATA_PATH is None:
    raise Exception('set your dataset path under DATA_PATH configuration.py')

if HOME_DIR is None:
    raise Exception('set your dataset path under HOME_DIR configuration.py')
