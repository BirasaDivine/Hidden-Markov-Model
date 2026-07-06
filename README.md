# Human Activity Recognition with a Hidden Markov Model

Recognizing four physical activities — standing, walking, jumping, still —
from smartphone accelerometer, gyroscope, and gravity data using a Gaussian
Hidden Markov Model (`hmmlearn`), trained with Baum-Welch and decoded with
Viterbi.

## Project structure

```
data/
  raw/          labelled recordings, one activity per file (50 Hz and 100 Hz)
  processed/    pickled harmonized sessions, features, trained model
src/
  preprocess.py   loads raw clips, harmonizes to a common 50 Hz grid, splits
                  off an unseen test set, windows recordings, plots samples
  features.py     extracts time- and frequency-domain features per window
  hmm.py          Gaussian HMM: model selection, Baum-Welch training, Viterbi
  evaluate.py     decodes held-out test sessions, reports metrics and plots
plots/          generated figures (signals, transition matrix, convergence,
                confusion matrix, emission means, decoded timeline)
har_hmm.ipynb   end-to-end notebook: data, features, model, evaluation, discussion
```

## Setup

```
pip install pandas numpy scipy scikit-learn hmmlearn matplotlib jupyter
```

## Running the pipeline

`har_hmm.ipynb` runs everything end-to-end and is the primary deliverable.
Each stage can also be run standalone from the command line:

```
python src/preprocess.py --raw data/raw --out data/processed --plots plots
python src/features.py --proc data/processed
python src/hmm.py --proc data/processed --plots plots
python src/evaluate.py --proc data/processed --plots plots
```

## Method summary

- **Data**: 54 labelled clips (standing, walking, jumping, still) recorded
  at 50 Hz or 100 Hz. 2 clips per activity (8 total) are held out, untouched
  during training, as the unseen test set.
- **Preprocessing**: 100 Hz clips are decimated to a common 50 Hz grid, then
  segmented into 2-second, 100-sample windows with 50% overlap.
- **Features**: 21 features per window — time-domain (RMS, std, min/max,
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

## Data note

Due to time constraints, the raw recordings were provided by a colleague
rather than personally collected. All preprocessing, feature engineering,
modeling, and analysis are original work — see the report for the full
disclosure.
