# Cycle Sharing
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

### My biggest Mistakes

- [ ] Panicked and tried everything to get model running
   - did not document 
- [ ] Spend almost all time with model instead of data
- [ ] Spend too much time with other datasets 

### Call with Paulo on March 28th 11:30

Things to discuss 
- [x] What are the requirements to become junior, senior, lead, director data scientist?
- [x] From your point of view, what do you think were the biggest mistakes I made?
  - You did not spend time with the data. 
  - no real future engineering
    - is it better to measure in minutes, hours, etc
    - did not use weather data, concatenate with location, make it more precise
  - You have to justify and explain why you are using a dense layer, why dropout, make the customer understand your decisions
- [x] How would you have solved the task?
    - see task.html file Paulo send
- [ ] In your opinion, what are the most important frameworks for ML?
- [x] How much experience with SQL is required?
    - You should really brush up on that
- [x] my next steps should be?
    - How to show quality of a model besides looking at mse or any other metric. Look at regression, segmentation, and classification tasks separately and find metrics to 
    measure a models quality
        - what are the standard libraries, look at Paulo's task.html to understand which tools he used and why
    - get familiar with all the ML models, learn pros and cons, their frameworks