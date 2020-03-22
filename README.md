# Deep-SVDD
This is my Minimal Pytorch implementation of [Deep One-Class Classification](http://data.bit.uni-bonn.de/publications/ICML2018.pdf) (Deep SVDD) for Anomaly Detection (ICML 2018).



# Results
This implementation achieves similar results as the original implementation provided by the authors.


| Inlier class     | [Original Pytorch Implementation ](https://github.com/lukasruff/Deep-SVDD-PyTorch) | This implementation  |
| ------------- |:-------------:| :-------------:|
| 0 | 97.77 ± 0.51 | 97.86 ± 0.67 |
| 1 | 99.42 ± 0.06 | 99.47 ± 0.01 |
| 2 | 88.91 ± 0.99 | 90.42 ± 1.70 |
| 3 | 90.5 ± 1.48 | 91.05 ± 0.27 |
| 4 | 93.48 ± 1.24 | 93.38 ± 0.93 |
| 5 | 86.01 ± 1.30 | 88.21 ± 1.21 |
| 6 | 98.27 ± 0.2 | 98.18 ± 0.1 |
| 7 | 95.07 ± 0.72 | 95.16 ± 0.67 |
| 8 | 92.99 ± 0.75 | 93.03 ± 0.12 |
| 9 | 95.77 ± 0.34 | 95.49 ± 0.06 |
