# Linear model

## please refer to src3
aim: given X, ouput Y

lm: score(X, i, y_i) = W * F(X, i, y_i)

## overall instruction
+ the W is actually the model parameters, the length of W equals to len(epsilon).
+ we first initialize W=0
+ if the predicted-tag is not the correct tag, update W 
  + all features parameters of the wrong tag (after instantialize) that in **epsilon** subtract 1;
  + all faetures parameters of the correct tag that in **epsilon** plus 1


## instantialize

## create feature space

## online training

```commandline
iterator 19
---training set---
num correctly-predict sent: 0.6102117061021171, acc over all tag: 0.9780971937029432
---dev set---
num correctly-predict sent: 0.10785340314136126, acc over all tag: 0.8430612691031221
---online training over---

```
