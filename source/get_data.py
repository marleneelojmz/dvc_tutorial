import dvc.api
import yaml

params = yaml.safe_load(open("params.yaml"))["data_acquisition"]

dvc.api.read(
    params['path'],
    repo = params['repo'],
    #mode='r'
)

