# Treenet-small
In this Project I have tested out a very branched architecture for image generation. The architecture is Fully convolutional, meaning that it can be used with input images of every size.

# The Task
The image generation task given to the network is very similar to Laplacian2art project, however there are three important differences.
1. The training here is non-adversarial, meaning that the loss function is predefined and does not constitute of another network attempting to discriminate between the generated and real images.
2. The dramatic reduction in the number of parameters in the model. The treenet architecture used here has only about 2.9 million parameters versus the approximately 32 million parameters in the Laplacian2art's generator model.
3. **No Skip Connections** The branched architecture and it's modular form (treenet blocks) allow for a sufficiently large gradient to be passed even to the deeper layers. This relaxes the requirement of having to add skip connections.

# Characteristics
1. Due to computation and hardware constraints, the training was done on 128x128 images of paintings. However as was already stated, the architecture is Fully Convolutional and modular and hence could be easily extended for larger images. In fact the same is being currently done on the cloud by me.

# To-do
1. Adding the results.
