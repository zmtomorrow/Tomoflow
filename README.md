# Tomoflow
A toy CNN library written in numpy.

This is originally written for the coursework of Advanced Deep Learning and Reinforcement Learning.

I find it's very interesting to write as a toy framework, which is flexible for the coursework purpose but not flexible for the general usage.

If I was born 5 years earlier, this maybe more popular! 

I will refine this framework if I have time, but the likelihood is small.

Example:
```
### Define Model:
model=NN(structure=['conv_maxpool', 'conv_maxpool', 'linear_relu', 'linear'])

### Train:
for iteration:
  model.train(batch_x, batch_y, learning_rate)
    
### Evaluation:
model.predict(x_test)
```




