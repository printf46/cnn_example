# A CNN Example
Convolutional neural networks are hierarchies of filters that exploit the fact that neighboring pixels in images are usually semantically related and distance pixels and images are usually not semantically related so you can just by grouping the pixels that are next to each other hierarchically together reconstruct the shape of objects and this is an important prior that we built into these models so they can converge quickly (Joscha Bach)

Example CNN code from here https://www.kaggle.com/code/pranjalsoni17/natural-scene-classification/notebook#Model-Fitting with debug added to confirm the
next layer size calculations as described here https://www.coursera.org/learn/convolutional-neural-networks/lecture/nsiuW/one-layer-of-a-convolutional-network (15:00)

i.e. n_height_curr_layer = floor((n_height_previous_layer + 2*padding_curr_layer - filter_size_curr_layer / stride_curr_layer) + 1)

(same formula for n_width_curr_layer)

See comments in NaturalSceneClassification class for Tensor sizes produced when running the notebook in Google Colab (GPU enabled)

Note that Pytorch equates kernel_size with filter_size as can be seen here https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

Overview of how the original CNN code works is here https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48 

