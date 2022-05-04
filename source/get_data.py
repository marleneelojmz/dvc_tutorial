import dvc.api

dvc.api.read(
    'tutorials/versioning/data.zip',
    repo = 'https://github.com/iterative/dataset-registry',
    #mode='r'
)

