# FACE RECOGNITION USING COSINE SIMILARITY

## Create a Virtual Environment

To create a virtual environment using Conda, run the following command:
NOTE: YOU NEED CUDA VERSION:12.4 TO BE COMPATIBLE WITH THE FOLLOWING DEPENDENCIES
```sh
conda env create -f environment.yml
```

To install the dependencies(run it if you need only the dependencies for the environment you already created)
```sh
pip install -r requirements.txt
```

To activate the environment
```sh
conda activate env_name
```
Replace env_name with your environment name

To run the code the dataset should be in the following format

```sh
dataset
 |-student-name1
 |   |-images.jpg
 |-student-name2
 |   |-images.jpg
 |-student-name3
 |   |-images.jpg
 .
 .
 .
 .
 ```

To run live_detect.py
```sh
python live_detect.py --dataset dataset_path --source "http://ip-address of the camera:port/video"
```


