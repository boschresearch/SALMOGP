# Safe Active Learning for Multi-Output Gaussian Processes

This is the companion code for the AISTATS 2022 paper [Safe Active Learning for Multi-Output Gaussian Processes](https://). The code allows the users to reproduce and extend the results reported in the study. Please cite the above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication [SAL for MOGP](https://). It will neither be maintained nor monitored in any way.
The general goal is to perform safe active learning on multi-output regression data. The data set comprise (x, y, z), which are input, output, and safety output. The variable z may or may not be one of the components of y. The output y can be obtained asynchronously (e.g. y=(nan, 0.2, nan, nan)).

## Setup

Clone this repository
```buildoutcfg
git clone https://github.com/boschresearch/SALMOGP.git
cd SALMOGP
```

Install minimal set of requirements
```
conda env create --file environment.yml
conda activate salmogp
```

## Experiment setting

All the setting we used are in [default_parameters.py](default_parameters.py). This script has 3 objects, each of which corresponds one experiment performed in the paper.

## Datasets

The paper perform experiments with 3 datasets, i.e. toy data, GP data, and OWEO data (EngE in the paper).

-------

The toy dataset is generated with numpy objects, and there is no need to do anything in advance.

-------

To generate GP dataset with default setting, run:
```
python 0_GP_data_sampler.py
```
Afterwards, the data description can be checked here [data/GP_samples/data_parameters.txt](data/GP_samples/data_parameters.txt). Notice that the safety values' distribution is dependent on the seed, and the parameter safety_threshold (<img src="https://render.githubusercontent.com/render/math?math=z_{bar}"> in the paper) in the experiment should be set accordingly.

To see all parser arguments, run:
```
python 0_GP_data_sampler.py --help
```

-------

The raw data are provided in [Bosch-Engine-Datasets/gengine1](https://github.com/boschresearch/Bosch-Engine-Datasets/tree/master/gengine1). Copy the raw csv files, put them into [data](data), and run the followings to obtain the OWEO data taken by the experiment scripts:
```
python 0_load_csv_addNXstructures.py --mode training
python 0_load_csv_addNXstructures.py --mode test
```
The script [0_load_csv_addNXstructures.py](0_load_csv_addNXstructures.py) incorporates history structure (this data set is dynamic), and we recommend not to change the default setting except for data/file paths.

-------

## Experiments

For all of the following scripts, use `--help` to access possible parser arguments. The datasets are not large and a GPU is not necessary unless we compute predictions on large sets (e.g. actively search on full OWEO dataset).
The current HMC implementation with gpflow & tensorflow_probability does not release the memory properly after each active learning iteration. Therefore, when we run the pipeline with Bayesian treatment, the more iterations we have, the more memory is consumed. To run huge number of iterations (for example more than 50 iterations might be too much for a laptop), one approach is to terminate the script after some iterations, save all of the data and parameters, and initiate the script again with these saved files. This is unfortunately not supported yet.

`safety_prob_threshold` here is the <img src="https://render.githubusercontent.com/render/math?math=1-\delta"> in the paper.
`safety_threshold` here is the <img src="https://render.githubusercontent.com/render/math?math=z_{bar}"> in the paper.

-------

toy, run:
```
python experiment_toy.py --POO True --num_init_data 12 --iteration_num 40 --fullGP True --bayesian True --safety_prob_threshold 0.95 --safety_threshold 0.7
```
Output y can be set to be observed synchronously `--POO False`.
After the experiment is finished, obtain the plots by running:
```
python experiment_toy_after_process.py
python afterprocess_safety_summary.py --Z_threshold 0.7 --safe_above_threshold True --dir ./experimental_result/toy*/*/
```

-------

GP data, run:
```
python experiment_GPsamples.py --POO True --num_init_data 40 --iteration_num 40 --fullGP True --bayesian True --safety_prob_threshold 0.95 --safety_threshold -0.56 
```
Output y can be set to be observed synchronously `--POO False`.
After the experiment is finished, obtain the plots by running:
```
python experiment_GPsamples_after_process.py 
python afterprocess_safety_summary.py --Z_threshold -0.56 --safe_above_threshold True --dir ./experimental_result/GPdata/X2L3Y4*/*/
```

-------

OWEO data (EngE in the paper), run:
```
python experiment_SAL_POO.py --num_init_data 40 --iteration_num 40 --fullGP True --bayesian True --safety_prob_threshold 0.95 --safety_threshold 1
```
It is possible to run the pipeline on fully observed outputs with script `experiment_SAL.py`. The parser arguments are the same.
We could add `--preselect_data_num 9000` to perform the experiment on a random subset of 9000 points, which is typically quite enough.
In this dataset, different outputs usually have different optimal history structures, but we could run the experimens on other output channels anyway with the same structures. `--used_y_ind id1 id2 ...`, id ranging from 0 to 7, specifies the indices of output we would like to model (on the already processed data).

After the experiment is finished, obtain the plots by running:
```
python experiment_GPsamples_after_process.py
python afterprocess_safety_summary.py --Z_threshold 1.0 --safe_above_threshold False --dir ./experimental_result/OWEO_HC_O2*/*/AL_n0_*/
```


-------

## License

SALMOGP is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in SALMOGP, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
