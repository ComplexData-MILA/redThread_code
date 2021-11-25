# RedThread Implementation

This repository contains the implementation for the RedThread algorithm from [this paper](https://dl.acm.org/doi/abs/10.1145/3219819.3220103). 

1. There are 2 non-trafficking related datasets used in the experiments [Discogs](http://data.discogs.com/?prefix=data/2019/) and [Memetracker](http://memetracker.org/data/index.html#twitter) which can be downloaded from their respective websites. The trafficking data can be shared separately via a google drive link. 

2. Once you have the data file, to first obtain the data features in the required format, run the code in the [`exploratory_analysis.ipynb`](./exploratory_analysis.ipynb) file. This code will create the sample data files inside `data` folder. 

3. Following this, run [`redthread_run.py`](./redthread_run.py) to run the RedThread algorithm. The default data path is set to [`data/sample_data/`](./data/sample_data) which points to the files created in Step 2. In order to use another path, provide the path as a command-line argument. To see other arguments, run `python redthread_run.py --help`. The current code runs very slowly and needs to be improved for efficiency. Due to the large size of the data, they have been uploaded [here](https://drive.google.com/drive/folders/164yhOadIogXCdns2q3jreVS09HrjhCIq?usp=sharing)

4. The graph models built after Step 3 will be stored in the [`models`](./models) folder. 


