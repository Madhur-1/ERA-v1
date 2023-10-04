# Session 17

## Introduction

This assignment focuses on getting more acquainted with the constituents of the transformer - encoder & the decode.

### Target
The target is to define a single transformer class that has the capability to be both an encoder and a decoder - i.e. the same class be used to train BERT, GPT & ViT models.

### Metrics
#### BERT

10000 it: 4.3 Loss

MLM task where the model is to predict the masked tokens


#### GPT

step 500 | train loss 0.2291 | val loss 7.0340

Next token prediction task

#### ViT

Epoch: 42 | train_loss: 1.1145 | train_acc: 0.2734 | test_loss: 1.0538 | test_acc: 0.5417

Image Multiclass classification task

## Training Log

### BERT
```
it: 9000  | loss 4.39  | Δw: 11.509
it: 9010  | loss 4.47  | Δw: 12.076
it: 9020  | loss 4.43  | Δw: 12.514
it: 9030  | loss 4.35  | Δw: 11.198
it: 9040  | loss 4.44  | Δw: 12.531
it: 9050  | loss 4.48  | Δw: 11.5
it: 9060  | loss 4.42  | Δw: 11.833
it: 9070  | loss 4.46  | Δw: 12.736
it: 9080  | loss 4.44  | Δw: 12.165
it: 9090  | loss 4.44  | Δw: 12.193
it: 9100  | loss 4.48  | Δw: 11.598
it: 9110  | loss 4.37  | Δw: 12.063
it: 9120  | loss 4.51  | Δw: 11.885
it: 9130  | loss 4.45  | Δw: 12.353
it: 9140  | loss 4.43  | Δw: 11.85
it: 9150  | loss 4.39  | Δw: 12.269
it: 9160  | loss 4.39  | Δw: 11.942
it: 9170  | loss 4.36  | Δw: 12.306
it: 9180  | loss 4.4  | Δw: 12.183
it: 9190  | loss 4.48  | Δw: 12.15
it: 9200  | loss 4.42  | Δw: 12.272
it: 9210  | loss 4.42  | Δw: 12.253
it: 9220  | loss 4.46  | Δw: 13.418
it: 9230  | loss 4.4  | Δw: 12.831
it: 9240  | loss 4.39  | Δw: 12.136
it: 9250  | loss 4.38  | Δw: 13.013
it: 9260  | loss 4.37  | Δw: 12.107
it: 9270  | loss 4.44  | Δw: 12.382
it: 9280  | loss 4.32  | Δw: 12.364
it: 9290  | loss 4.47  | Δw: 11.497
it: 9300  | loss 4.34  | Δw: 11.932
it: 9310  | loss 4.39  | Δw: 12.593
it: 9320  | loss 4.46  | Δw: 12.455
it: 9330  | loss 4.39  | Δw: 12.459
it: 9340  | loss 4.33  | Δw: 12.311
it: 9350  | loss 4.35  | Δw: 12.406
it: 9360  | loss 4.51  | Δw: 12.365
it: 9370  | loss 4.51  | Δw: 12.074
it: 9380  | loss 4.47  | Δw: 11.953
it: 9390  | loss 4.46  | Δw: 12.269
it: 9400  | loss 4.17  | Δw: 12.16
it: 9410  | loss 4.38  | Δw: 11.511
it: 9420  | loss 4.18  | Δw: 12.128
it: 9430  | loss 4.36  | Δw: 11.96
it: 9440  | loss 4.4  | Δw: 12.873
it: 9450  | loss 4.38  | Δw: 12.159
it: 9460  | loss 4.58  | Δw: 13.319
it: 9470  | loss 4.43  | Δw: 12.513
it: 9480  | loss 4.55  | Δw: 11.999
it: 9490  | loss 4.28  | Δw: 12.732
it: 9500  | loss 4.34  | Δw: 12.097
it: 9510  | loss 4.46  | Δw: 12.751
it: 9520  | loss 4.49  | Δw: 12.123
it: 9530  | loss 4.32  | Δw: 12.09
it: 9540  | loss 4.5  | Δw: 12.6
it: 9550  | loss 4.38  | Δw: 12.654
it: 9560  | loss 4.53  | Δw: 12.361
it: 9570  | loss 4.37  | Δw: 12.642
it: 9580  | loss 4.41  | Δw: 12.462
it: 9590  | loss 4.53  | Δw: 12.512
it: 9600  | loss 4.46  | Δw: 13.208
it: 9610  | loss 4.28  | Δw: 13.396
it: 9620  | loss 4.36  | Δw: 12.489
it: 9630  | loss 4.28  | Δw: 12.354
it: 9640  | loss 4.38  | Δw: 12.494
it: 9650  | loss 4.45  | Δw: 12.74
it: 9660  | loss 4.33  | Δw: 13.338
it: 9670  | loss 4.31  | Δw: 11.995
it: 9680  | loss 4.33  | Δw: 12.443
it: 9690  | loss 4.48  | Δw: 12.529
it: 9700  | loss 4.27  | Δw: 13.019
it: 9710  | loss 4.51  | Δw: 12.768
it: 9720  | loss 4.45  | Δw: 12.266
it: 9730  | loss 4.33  | Δw: 12.358
it: 9740  | loss 4.45  | Δw: 13.142
it: 9750  | loss 4.33  | Δw: 12.587
it: 9760  | loss 4.49  | Δw: 12.97
it: 9770  | loss 4.4  | Δw: 12.928
it: 9780  | loss 4.44  | Δw: 12.851
it: 9790  | loss 4.24  | Δw: 12.528
it: 9800  | loss 4.37  | Δw: 12.896
it: 9810  | loss 4.3  | Δw: 12.968
it: 9820  | loss 4.41  | Δw: 12.912
it: 9830  | loss 4.38  | Δw: 12.933
it: 9840  | loss 4.38  | Δw: 12.847
it: 9850  | loss 4.19  | Δw: 12.809
it: 9860  | loss 4.36  | Δw: 12.819
it: 9870  | loss 4.3  | Δw: 12.916
it: 9880  | loss 4.3  | Δw: 13.15
it: 9890  | loss 4.52  | Δw: 12.605
it: 9900  | loss 4.37  | Δw: 12.817
it: 9910  | loss 4.45  | Δw: 12.639
it: 9920  | loss 4.41  | Δw: 13.198
it: 9930  | loss 4.29  | Δw: 13.123
it: 9940  | loss 4.39  | Δw: 13.814
it: 9950  | loss 4.39  | Δw: 12.96
it: 9960  | loss 4.3  | Δw: 12.604
it: 9970  | loss 4.33  | Δw: 13.448
it: 9980  | loss 4.44  | Δw: 12.589
it: 9990  | loss 4.3  | Δw: 13.889
```

### GPT
```
step        400 | loss 0.5148
step        401 | loss 0.5744
step        402 | loss 0.5668
step        403 | loss 0.5224
step        404 | loss 0.4652
step        405 | loss 0.4447
step        406 | loss 0.4686
step        407 | loss 0.4885
step        408 | loss 0.5444
step        409 | loss 0.4380
step        410 | loss 0.5071
step        411 | loss 0.4489
step        412 | loss 0.4985
step        413 | loss 0.4810
step        414 | loss 0.5410
step        415 | loss 0.4693
step        416 | loss 0.4532
step        417 | loss 0.4703
step        418 | loss 0.4330
step        419 | loss 0.4134
step        420 | loss 0.4598
step        421 | loss 0.4652
step        422 | loss 0.4835
step        423 | loss 0.4332
step        424 | loss 0.4238
step        425 | loss 0.4313
step        426 | loss 0.4541
step        427 | loss 0.4654
step        428 | loss 0.4881
step        429 | loss 0.4452
step        430 | loss 0.4548
step        431 | loss 0.4165
step        432 | loss 0.4027
step        433 | loss 0.4202
step        434 | loss 0.3811
step        435 | loss 0.3763
step        436 | loss 0.4419
step        437 | loss 0.4716
step        438 | loss 0.4290
step        439 | loss 0.4053
step        440 | loss 0.4019
step        441 | loss 0.3829
step        442 | loss 0.4836
step        443 | loss 0.3957
step        444 | loss 0.4427
step        445 | loss 0.4602
step        446 | loss 0.4059
step        447 | loss 0.4285
step        448 | loss 0.3779
step        449 | loss 0.4175
step        450 | loss 0.4418
step        451 | loss 0.3834
step        452 | loss 0.4341
step        453 | loss 0.4537
step        454 | loss 0.3371
step        455 | loss 0.3931
step        456 | loss 0.3747
step        457 | loss 0.3936
step        458 | loss 0.3463
step        459 | loss 0.4057
step        460 | loss 0.3425
step        461 | loss 0.3832
step        462 | loss 0.3830
step        463 | loss 0.4011
step        464 | loss 0.3936
step        465 | loss 0.3821
step        466 | loss 0.4207
step        467 | loss 0.3564
step        468 | loss 0.3219
step        469 | loss 0.3447
step        470 | loss 0.3577
step        471 | loss 0.3370
step        472 | loss 0.4545
step        473 | loss 0.3397
step        474 | loss 0.3321
step        475 | loss 0.3419
step        476 | loss 0.3399
step        477 | loss 0.3936
step        478 | loss 0.3481
step        479 | loss 0.3673
step        480 | loss 0.3483
step        481 | loss 0.3479
step        482 | loss 0.3520
step        483 | loss 0.3587
step        484 | loss 0.3433
step        485 | loss 0.3411
step        486 | loss 0.3224
step        487 | loss 0.3393
step        488 | loss 0.3368
step        489 | loss 0.3672
step        490 | loss 0.3318
step        491 | loss 0.3594
step        492 | loss 0.3754
step        493 | loss 0.3052
step        494 | loss 0.3041
step        495 | loss 0.3151
step        496 | loss 0.3541
step        497 | loss 0.3129
step        498 | loss 0.3089
step        499 | train loss 0.2291 | val loss 7.0340
step        499 | loss 0.3225
```

### ViT
```
Epoch: 20 | train_loss: 1.1507 | train_acc: 0.2930 | test_loss: 1.2421 | test_acc: 0.1979
Epoch: 21 | train_loss: 1.0846 | train_acc: 0.4453 | test_loss: 1.1815 | test_acc: 0.2604
Epoch: 22 | train_loss: 1.1038 | train_acc: 0.4258 | test_loss: 1.1374 | test_acc: 0.2604
Epoch: 23 | train_loss: 1.1343 | train_acc: 0.3047 | test_loss: 1.1025 | test_acc: 0.2604
Epoch: 24 | train_loss: 1.1344 | train_acc: 0.2812 | test_loss: 1.0378 | test_acc: 0.5417
Epoch: 25 | train_loss: 1.1292 | train_acc: 0.2891 | test_loss: 1.1337 | test_acc: 0.2604
Epoch: 26 | train_loss: 1.0929 | train_acc: 0.4297 | test_loss: 1.0573 | test_acc: 0.5417
Epoch: 27 | train_loss: 1.1268 | train_acc: 0.2812 | test_loss: 1.0592 | test_acc: 0.5417
Epoch: 28 | train_loss: 1.1325 | train_acc: 0.3047 | test_loss: 1.1948 | test_acc: 0.1979
Epoch: 29 | train_loss: 1.1569 | train_acc: 0.2969 | test_loss: 1.1696 | test_acc: 0.2604
Epoch: 30 | train_loss: 1.0886 | train_acc: 0.4414 | test_loss: 1.0279 | test_acc: 0.5417
Epoch: 31 | train_loss: 1.1584 | train_acc: 0.2812 | test_loss: 1.0452 | test_acc: 0.5417
Epoch: 32 | train_loss: 1.0817 | train_acc: 0.4023 | test_loss: 1.2420 | test_acc: 0.1979
Epoch: 33 | train_loss: 1.1465 | train_acc: 0.2930 | test_loss: 1.2285 | test_acc: 0.1979
Epoch: 34 | train_loss: 1.1194 | train_acc: 0.3203 | test_loss: 1.1031 | test_acc: 0.2604
Epoch: 35 | train_loss: 1.0948 | train_acc: 0.3906 | test_loss: 1.0202 | test_acc: 0.5417
Epoch: 36 | train_loss: 1.1548 | train_acc: 0.2812 | test_loss: 1.0535 | test_acc: 0.5417
Epoch: 37 | train_loss: 1.1379 | train_acc: 0.3086 | test_loss: 1.1902 | test_acc: 0.1979
Epoch: 38 | train_loss: 1.1138 | train_acc: 0.2812 | test_loss: 1.0961 | test_acc: 0.1979
Epoch: 39 | train_loss: 1.0850 | train_acc: 0.4141 | test_loss: 1.1648 | test_acc: 0.1979
Epoch: 40 | train_loss: 1.0922 | train_acc: 0.4141 | test_loss: 1.1788 | test_acc: 0.1979
Epoch: 41 | train_loss: 1.1308 | train_acc: 0.2930 | test_loss: 1.1288 | test_acc: 0.1979
Epoch: 42 | train_loss: 1.1145 | train_acc: 0.2734 | test_loss: 1.0538 | test_acc: 0.5417
```
