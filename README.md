# continuousf0eval
Continuous Metrics for Evaluating Single-f0 Estimation

Companion code for the paper:


"Generalized Metrics for Single-F0 Estimation Evaluation"

Rachel M. Bittner and Juan José Bosch

in International Society for Music Information Retrieval (ISMIR) Conference, 2019


```
@inproceedings{bittner_bosch_2019, title={Generalized Metrics for Single-F0 Estimation Evaluation}, author={Bittner, Rachel M and Bosch, Juan J.}, booktitle={International Society for Music Information Retrieval (ISMIR) Conference}, year={2019}}
```


## Contents

### algorithm_outputs

The folder `algorithm_outputs` contains outputs of the following algorithms:

* [crepe](https://github.com/marl/crepe)
* [deep salience](https://github.com/rabitt/ismir2017-deepsalience/blob/master/predict/predict_on_audio.py)
* [melodia](https://www.upf.edu/web/mtg/melodia)
* [pyin](https://code.soundsoftware.ac.uk/projects/pyin)

on the datasets:

* [medleydb-pitch](https://zenodo.org/record/2620624#.XZ5HkedKhTY)
* [meldeydb-melody](https://zenodo.org/record/2628782#.XZ5HrOdKhTY)
* [ikala](http://mac.citi.sinica.edu.tw/ikala/)
* [orchset](https://zenodo.org/record/1289786#.XZ5Hv-dKhTY)

### experiments

The folder `experiments` contains the code we used to run the experiments.

* `confidence` : confidence estimates. `confidence/separation` contains the confidence computed on the ikala source separated vocals. `confidence/stems` contains the confidence computed on the clean ikala vocals and on the medleydb-pitch sources.

* `compute_confidence.py` : the script that was used to generate the confidence files in `confidence`

* `confidence.py` : a module for loading the confidence values for different algorithms & ground truth datasets.

* `Experiment Plots.ipynb` : notebook used to generate Figures 2-6 in the paper

* `metrics.py` : a module with the proposed metrics implemented

* `outputs.py` : a module for loading algorithm output files

* `plot.py` : a module with a few plotting utility functions

* `Plots toy examples.ipynb` : notebook used to generate Figure 1 in the paper

### paper-figs

Generated paper figures.
