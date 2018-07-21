## Meaning of STATUS in bureau_balance.csv

* "C"はclosedほんまか？

fitifox
> Hi guys,

> Do you know the meaning of STATUS = "C" in bureau_balance.csv

> THank you so much, Regards.

CoreyLevinson
> Hi Carlos. In my modek I am considering "closed" to mean the same thing as the value 0. Both I believe mean successful payment without being late.

HemantRupani
> Yes, both meant successful payment without being late. In "C"-the Account has been closed and in "0" - the Account is active at the time of application.

* STATUSが"C"でも、bureau.csvではActiveなものもあり

CREDIT_ACTIVE  CREDIT_ACTIVE
Active         Active            43480
Closed         Closed           283362
Sold           Sold                471
Name: CREDIT_ACTIVE, dtype: int64

* 参考までにbureau.csv全体では

CREDIT_ACTIVE  CREDIT_ACTIVE
Active         Active            630607
Bad debt       Bad debt              21
Closed         Closed           1079273
Sold           Sold                6527
Name: CREDIT_ACTIVE, dtype: int64
