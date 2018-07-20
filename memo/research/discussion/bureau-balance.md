## What does the bureau balance data mean?

### STATUSとBureau Balanceから見えること

* 過去の支払いの遅れ具合がわかる？

Luis Moneda
> Hi, I looked for further information at the "Data" section in here at the forum searching for "bureau balance", but I wasn't able to find anything to help me.

> From "Data" tab:
> Monthly balances of previous credits in Credit Bureau.
> This table has one row for each month of history of every previous credit reported to Credit Bureau - i.e the table has (#loans in sample * # of months where we have some history observable for the previous credits) rows.

> The #loans in the sample are about the Home Credit loans or the ones registered in the bureau data? For example, the `SK_ID_CURR` 215354 has 9 registers in bureau.csv, but when I try to find one of the `SK_ID_BUREAU` on the bureau_balance.csv table, there's no balance!

> Son I don't get the `(#loans in sample * # of relative previous credits * # of months where we have some history observable for previous credits)`.

> Who ends up having bureau_balance info ?
> Also, what does the STATUS mean? The unique values for it are:
> `['C', '0', 'X', '1', '2', '3', '4', '5']`

Russ Beuker
>The bureau_balance table can be a bit confusing. Here is my take on how to interpret it. I welcome your comments and observations!

>Let's look at some examples of loan payment histories. Think of how you would 'score' each loan. Low scores would push the TARGET towards zero. High scores would push TARGET towards one. You could devise a method to calculate an overall score for each loan in the bureau_balance table, based on how long ago something good/bad happened to the loan repayment.

>Data Definitions from the HomeCredit_columns_description.csv file:

>MONTHS_BALANCE -Month of balance relative to application date (-1 means the freshest balance date)

>STATUS - Status of Credit Bureau loan during the month (active, closed, DPD0-30,… [C means closed, X means status unknown, 0 means no DPD, 1 means maximal did during month between 1-30, 2 means DPD 31-60,… 5 means DPD 120+ or sold or written off ] )

>A good recent loan that should push TARGET towards zero.
>Actual data:
>    SK_ID_BUREAU    MONTHS_BALANCE  STATUS        MYCOMMENT
>    5001729         0               C             loan is paid off
>    5001729         -1              0             payment on time
>    5001729         -2              0             payment on time
>    5001729         -3              0             payment on time
>    5001729         -4              0             payment on time
>    5001729         -5              0             payment on time
>    5001729         -6              0             payment on time
>My score: 0 (means perfect)

>A not-so-great recent loan that should push TARGET towards one.
>Simulated data:
>    SK_ID_BUREAU    MONTHS_BALANCE  STATUS        MYCOMMENT
>    5001711         0               X             missing data.  Hmmm...
>    5001711         -1              0             payment on time
>    5001711         -2              0             payment on time
>    5001711         -3              0             payment on time
>    5001711         -4              1             payment late
>My score: 500 (shaky)

>A not-so-recent, not-so-great loan that should push TARGET towards one, though maybe not by as much if the loan was more recent.
>Actual data:
>    SK_ID_BUREAU    MONTHS_BALANCE  STATUS        MYCOMMENT
>    5001722         -38             C             loan is closed
>    5001722         -39             0             payment on time
>    5001722         -40             0             payment on time
>    5001722         -41             1             payment late
>    5001722         -42             1             payment late
>    5001722         -43             0             payment on time
>    5001722         -44             1             payment late
>    5001722         -45             1             payment late
>    5001722         -46             1             payment late
>My score: 300 (really shaky, but it was a long time ago)

>A really bad loan that should strongly push TARGET towards one. This is a loan default that was written off or sold to a collection agency.  Notice that a closed loan doesn't mean a good loan.
>Actual data:
>    SK_ID_BUREAU    MONTHS_BALANCE  STATUS        MYCOMMENT
>    5001797        -54              C             loan is closed
>    5001797        -55              5             loan defaulted
>    5001797        -56              5             payment late !!!!
>    5001797        -57              4             payment late !!!
>    5001797        -58              3             payment late !!
>    5001797        -59              2             payment late !
>    5001797        -60              1             payment late
>    5001797        -61              1             payment on late
>    5001797        -62              0             payment on time
>    5001797        -63              1             payment late
>    5001797        -64              0             payment on time
>    5001797        -65              1             payment late
>    5001797        -66              0             payment on time
>    5001797        -67              0             payment on time
>    5001797        -68              0             payment on time
>    5001797        -69              0             payment on time
>My score: 1000 (as bad as it gets, even though it was a long time ago.  Could only be worse if it defaulted last month instead of 55 months ago, which would score 10000)
>So now we have a single score per loan. I just guessed at the scores, so you'll need to come up with a formula for your own scores.

>Then we need to deal with an applicant having more than one loan history. Do we average the loan history scores? What do we do if there is a loan but no history at all?

>Tip: Consider making the score formula to be somehow part of your hyperparameter search.

>RNN's ? Since we kind of encoded a load payment history in the score, we may not need a RNN-like model to handle the temporal data. Just a thought.

>It's worth noting that there are some loan balance history data in bureau_balance for ID_SK_BUREAU's that don't exist in the bureau table. These probably can be ignored since they can't be correlated with anything else from the other tables.

>Late Edit: Seems that the some level of aggregation is expected, based on the competition info:

>https://www.kaggle.com/c/home-credit-default-risk/discussion/57054

>We have provided the data in the raw form (instead of aggregations of time variant data on loan level we provided the underlying data needed for such aggregation) as we are keen to not only see what modelling approaches will the Kaggle community use but also how will the community work with the data in such form.

>Have fun!
