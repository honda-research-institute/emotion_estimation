## HRI dataset:
- 9 subjects (7 initially and 2 in the 2nd batch) 
- There are two 30 min events marked by 1 and 2 inside the subject directory
- The start time of each picture displayed to the subject is noted in the [name]_pics.csv 
- Each event has data from 5 modalities : ECG, EMG, GSR, PPG, RSP 

### Processing: 
- Physiological data from each channel is standardized using the mean and standard deviation. This step is carried for all the subjects to remove individual differences 
- Some GSR signals had inf symbol in the data, this is replaced by zeros in the csv files
- Subject main-exp-6333-m has data missing from the last 5 min of event 2, the corresponding information is removed from s2_pics.csv


### EMG paper:
- Repeatability of facial electromyography (EMG) activity over corrugator supercilii and zygomaticus major on differentiating various emotions
