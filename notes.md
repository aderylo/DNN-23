## Lab 4:

Common mistakes with pytorch:
- not zeroing gradients after each iter; 
- numerical errors;
- vanishing/exploding gradients;


## Lab 5: 

 * SIFT: 
https://en.wikipedia.org/wiki/Scale-invariant_feature_transform 

* data -> features -> model -> $\hat{y}$

* vanishing gradient (in case of sigmoid):
    * vanishing: if you have $0.9^{51}$ then it will aproach $0.000000$ and finally reach zero when we run out of precision;
    * explosion: if you have $1.1^{51}$ then it will explode till inf when we run out of range; 

    So why sigmoids gradient vanishes? Well because its derivative looks like that: $s(x) (1 - s(x))$, so either sides of the product will approach zero if $x \in (-inf, -3) \cup (3, inf)$ i.e. disappear. 


#### how to deal with it? 

1. Weight initalization: Glorot initalization; \
2. Normalization;
3. Batch normalization:
$$
y = \frac{x - E(x)}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta
$$