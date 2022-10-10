Siamese Neural Network (SNN)
========
Classification model of the ADNI brain data[OASIS brain](https://adni.loni.usc.edu/) data set using [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

**Author:** *Lukas Lobmaier (s4736267)*

* [SNN Architecture](#SNN-Architecture)<br>
* [Design Approach](#Design-Approach)<br>
* [Executing Code](#Executing-Code)<br>
* [Results](#Results)<br>
* [References](#References)

## SNN Architecture

The main idea of a Siamese Neural Network (SNN) , sometimes also refered as twin nural net [1], is to compare two inputs regarding their similarity. Therefore, the SNN has a unique structure compared to other neural nets.  </br>
The main strcuture constist of two identicall sub nets, which are processing each of the two input data samples. The outputs of these subnets, can be refered as a complex feature mapping or fingerprint of the input sample, are then compared regarding similarity. 

<p align="center">
    <img src="Pictures/OverviewSNN.png" width="600" >
</p>
<p align="center">
    <em> Figure 1: Overview of SNN approach  </em>
</p>
https://en.wikipedia.org/wiki/Animal

The output of a classical SNN returns a single value:
* Same:    &nbsp;&nbsp;&nbsp;&emsp;Input samples are from the same class 
* Different:  &nbsp;Input samples are from different classes.

After the network has been tuned, the SNN provides good discriminative features. Due to this, the SNN can then be used to predict classes of new data. Furtheremore, it can also be used on different, new classes which have not been seen by the network during the training procedure.

Zu den Anwendungen von Ähnlichkeitsmaßen, bei denen ein Zwillingsnetz zum Einsatz kommen kann, gehören beispielsweise die Erkennung handgeschriebener Schecks, die automatische Erkennung von Gesichtern in Kamerabildern und der Abgleich von Abfragen mit indizierten Dokumenten. Die vielleicht bekannteste Anwendung von Zwillingsnetzen ist die Gesichtserkennung, bei der bekannte Bilder von Personen vorberechnet und mit einem Bild von einem Drehkreuz oder ähnlichem verglichen werden. Es ist auf den ersten Blick nicht offensichtlich, aber es gibt zwei leicht unterschiedliche Probleme. Das eine ist das Erkennen einer Person unter einer großen Anzahl anderer Personen, das ist das Gesichtserkennungsproblem. DeepFace ist ein Beispiel für ein solches System,[4] in seiner extremsten Form ist dies die Erkennung einer einzelnen Person auf einem Bahnhof oder Flughafen. Die andere ist die Gesichtsverifikation, d. h. die Überprüfung, ob das Foto in einem Ausweis mit der Person übereinstimmt, die behauptet, dieselbe Person zu sein. Das Zwillingsnetz mag dasselbe sein, aber die Umsetzung kann sehr unterschiedlich sein.

https://towardsdatascience.com/what-are-siamese-neural-networks-in-deep-learning-bb092f749dcb


we can then capitalize on powerful discriminative features to generalize the predictive power of
the network not just to new data, but to entirely
new classes from unknown distributions. Using a
convolutional architecture, we are able to achieve
strong results which exceed those of other deep
learning models with near state-of-the-art performance on one-shot classification tasks.



 Häufig wird einer der Ausgangsvektoren vorberechnet und bildet so eine Basislinie, mit der der andere Ausgangsvektor verglichen wird. Dies ist vergleichbar mit dem Vergleich von Fingerabdrücken, kann aber technisch als Distanzfunktion für ortsabhängiges Hashing beschrieben werden.

## Report's StyleGAN Design 
## Design Approach
## Executing Code
## Results
## References
* https://adni.loni.usc.edu/
* https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
* https://en.wikipedia.org/wiki/Siamese_neural_network