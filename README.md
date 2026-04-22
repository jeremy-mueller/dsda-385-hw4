# DSDA 385 - Homework 4
## Introduction
The objective of this assignment is to learn how to apply NLP techniques while working with a large-scale real-world dataset. We should also learn more about multi-headed self-attention architecture and communicate our findings.

## Dataset
Our dataset is the Microsoft News Dataset which contains 65,000 news articles, 230,000 impressions, and 50,000 users. Our goal is to create a recommendation system to optimize the articles suggested to readers.

## Architecture
We will implement a multi-headed self-attention with a news encoder that converts articles into vectors, and user encoder that turns a user's clicked article history into a vector. The dot product of those two vectors gives a score for how well suited the article is for the user.

## Hyperparameters
I've trained using three different combinations of hyperparameters, which I've labeled Hyper [1-3]. The breakdowns are as follows:
|Model|Learning Rate|Batch Size|Negative Samples|Attention Heads|Dropout|Max History Length|
|-|-|-|-|-|-|-|
|Hyper 1|1e-4|64|4|16|0.2|50|
|Hyper 2|1e-4|64|8|16|0.3|100|
|Hyper 3|1e-4|64|4|20|0.2|50|

The learning rate is also scheduled to decay by half every two epochs.

## Results
|Model|Loss|AUC|MRR|nDCG@5|nDCG@10|
|-|-|-|-|-|-|
|Hyper 1|1.3409|0.4776|0.4509|0.5847|0.5847|
|Hyper 2|1.3459|0.4809|0.4532|0.5865|0.5865|
|Hyper 3|1.3162|0.4806|0.4557|0.5883|0.5883|

Comparing this to the baselines in the assignment guide, our AUC scores are all worse than random, but MRR, nDCG@5, and nDCG@10 all greatly outperform even the tuned NRMS. Additionally, nDCG@5 and nDCG@10 are equal across all 3 of our hyperparameter experiments.

We can also see evidence of the loss converging throughout training, as shown in this graph below:

![](/results/hyperparameter_comparison.png)

## Comparison
Hyper 3 had the best overall results, which is likely due to the fact that it just had more attention heads and was able to make better connections between the users and articles. Hyper 2 had better results on average than Hyper 1 likely due to the higher negative sampling and max history length.

## Conclusion
I think this project helped a lot with my understanding of language processing as well as recommendation systems. It would be fun to see how a system like this could work in fields other than just news such as social media posts, movies, or video games.

I would like to see in the future how this architecture would work with different combinations of hyperparameters, but I have run out of time. For example, with a higher negative sampling, max history length, and amount of attention heads.

Results could also be improved by, obviously, using MINDlarge instead as it would have more training data and be able to create more accurate vectors for the articles and users.
