# LinearChain CRF
# model
crf04/crf05
# run
t4/t5
# notes
## viterbi/forward
if current timestep is t,  
the two algorithms will argmax/logsumexp the t-1 dim
## backward
if current timestep is t,  
the backward algorithm will logsumexp the t+1 dim

# result
```commandline

DataLoader(
  num of sentences: 803
  num of uniq words: 4537
  num of uniq tags: 31
  num of uniq chars: 1896
)
DataLoader(
  num of sentences: 1910
  num of uniq words: 8577
  num of uniq tags: 30
  num of uniq chars: 2524
)
---feature space constructing over---
---training set---
acc: 0.9250024445096313
---dev set---
acc: 0.8170472386176196
---online training over---
```