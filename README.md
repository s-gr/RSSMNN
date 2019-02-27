# Rossman Data Challange 

## Data set 
You can find the data set provided and used by fast.ai [here](http://files.fast.ai/part2/lesson14/rossmann.tgz). We migrate the notebook created by Jeremy P. Howard for the fast
.ai course on Machine Larning to Keras. This was originally presented in PyTorch using the fast.ai package. We start with a short overview of the data set and explain quickly 
what kind of information is given in them.

### TODOs

- [x] Create list of DataFrames from provided csv files
- [ ] Create summary to get feel for the data provided by the costumer
  - [ ] what data is missing
  - [x] show basic info:
    - [x] count
    - [x] mean
    - [x] std
    - [x] min
    - [x] quantiles
    - [x] max
    - [x] uniques
    - [x] types
- [ ] Think about visualizing the data?
- [ ] What is the story?

### Data Cleaning and Feature Engineering
Since we are handling tabular data, we have to do all the necessary cleaning and feature engineering steps. 