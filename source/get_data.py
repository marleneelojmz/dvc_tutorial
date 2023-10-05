import os
import sys
import argparse
from typing import Text
import subprocess
import dvc.api
import yaml

# this line was included for git executation errors
os.environ["GIT_PYTHON_REFRESH"] = "quiet"


def get_data(config_path: Text) -> None:

    params = yaml.safe_load(open(config_path))["data_acquisition"]
    print(params)
    
    if not os.path.isdir("data.zip"):
        subprocess.run(f"dvc get {params['repo']} {params['path']}")
        print('Hecho !')
    else:
        print('the folder alrready exist')

    '''dvc.api.read(
        params['path'],
        repo = params['repo'],
        mode='r'
        )'''
    
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--params', dest='params', required=True)
    args = args_parser.parse_args()

    get_data(config_path=args.params)


