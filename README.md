# Human Activity Recognition with a Hidden Markov Model

Recognizing four physical activities(standing, walking, jumping, still)
from smartphone accelerometer, gyroscope, and gravity data using a Gaussian
Hidden Markov Model (`hmmlearn`), trained with Baum-Welch and decoded with
Viterbi.


## Setup

```
pip install pandas numpy scipy scikit-learn hmmlearn matplotlib jupyter
```

## Running the pipeline

`hidden_activity.ipynb` runs everything end-to-end and is the primary deliverable.
Each stage can also be run standalone from the command line:

```
python src/preprocess.py --raw data/raw --out data/processed --plots plots
python src/features.py --proc data/processed
python src/hmm.py --proc data/processed --plots plots
python src/evaluate.py --proc data/processed --plots plots
```
## Project structure

```
data/
  raw/          labelled recordings, one activity per file (50 Hz and 100 Hz)
  processed/    pickled harmonized sessions, features, trained model
src/
  preprocess.py   
  features.py     
  hidden.py          
  evaluate.py     
figures/          generated figures 

hidden_activity.ipynb   notebook
```
## Method summary

- **Data**: 54 labelled clips (standing, walking, jumping, still) recorded
  at 50 Hz or 100 Hz. 2 clips per activity (8 total) are held out, untouched
  during training, as the unseen test set.
- **Preprocessing**: 100 Hz clips are decimated to a common 50 Hz grid, then
  segmented into 2-second, 100-sample windows with 50% overlap.
- **Features**: 21 features per window, time-domain (RMS, std, min/max,
  signal magnitude area, axis correlation), frequency-domain (dominant
  frequency, spectral energy, spectral entropy), and orientation (mean
  gravity-vector components, which separate postures like standing vs.
  still that look identical in motion alone).
- **Model**: a 4-state diagonal/full-covariance Gaussian HMM (chosen via a
  validation split carved from training data), trained unsupervised with
  Baum-Welch, with discovered states mapped to activity labels by majority
  vote on training data.
- **Evaluation**: Viterbi decoding on the 8 held-out unseen recordings,
  reporting per-activity sensitivity, specificity, and accuracy, alongside
  confusion matrix, transition matrix, emission means, and decoded-sequence
  timeline plots.

## Results

Overall accuracy on unseen data: **80.4%**. Walking, jumping, and still are
decoded almost perfectly; standing is the weak point, most often confused
with walking. Full discussion is in the notebook's final section and in the
written report.


