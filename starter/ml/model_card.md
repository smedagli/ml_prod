# Model card

## Model details

The model uses categorical and numerical features to predict whether the salary is <= or > 50k.

### Categorical features
* workclass
* education
* marital-status
* occupation
* relationship
* race
* sex
* native-country

### Numerical features
* age
* education_num
* capital_gain
* capital_loss
* fnlgt
* hours_per_week

### Label (to predict)
* salary
  * 0 --> salary <= 50k
  * 1 --> salary > 50k

## Intended use

### Model training
to train the model run
```commandline
python -m ml_prod.starter.train_model
```
the script will write the performance of the model (see details below) in the folder `ml_prod/data/performance`,
and will also save the feature encoders in `ml_prod/model`.

The trained model will be saved as `ml_prod/model/model.pkl`.


### Inference
TODO

## Dataset
The complete dataset can be found in `ml_prod/data/census.csv`

### Training data and Evaluation data
Training and test data split are generated with `from sklearn.model_selection import train_test_split`.  
See `starter.train_model.py` at line 21
```python
train_df, test_df = train_test_split(data_, test_size=0.20, random_state=42)  # split dataset
```

## Metrics
Model performance are evaluated via different metrics, mainly:
* accuracy
* recall (sensitivity)
* precision (positive predictive value)
* specificity
* f1 score
* negative predictive value
* true negatives (tn)
* true positives (tp)
* false negatives (fn)
* false positives (fp)

Details about model performance on train and test datasets are provided in `ml_prod/data/performance`

### Confusion matrices
`train_confusion_matrix.csv` and `test_confusion_matrix.csv` report the confusion matrices computed on train and test 
set.

To read a confusion matrix
```python
               true positives              false positives
               false negatives             true negatives
```
To obtain additional information about the confusion matrices,
use the class `BinaryConfusionMatrix` defined in `ml_prod.starter.common`
to easily the metrics defined in the previous section.

### Data slices
In `ml_prod/data/performance` are also provided performance computed on data slices.  
The standard filename is
```commandline
<dataset>_<feature>_slice.csv
```
where `<dataset>` can be `test` or `train`.  
`<feature>` represents the feature on which the data is sliced.
The file includes all the metrics computed on each category of the selected feature.

#### Example:
The feature `sex` includes categories `male` and `female`.
Here is the example file `test_sex_slice.csv`
```csv

    Category  accuracy  recall  precision  sensitivity  specificity  f1_score  negative_predictive_value  positive_predictive_value    tn   tp   fn  fp
0     female      0.90    0.29       0.53         0.29         0.97      0.37                       0.92                       0.53  1923   65  162  57
1       male      0.77    0.27       0.78         0.27         0.97      0.41                       0.76                       0.78  2959  343  909  95
```

## Known limitations
N/A

## Others
Other details can be found in the `notebooks` folder.