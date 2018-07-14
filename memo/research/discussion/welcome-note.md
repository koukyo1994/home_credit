## Welcome note from Home Credit

Home CreditのR&D部門

application, 統計データ, 過去の信用行動データから未来の行動に関して予測する。
モデリングのアプローチのみではなく、生の状態で渡されるデータに対してKaggleコミュニティがどのように動くのか
という部分もみる。完全なブラックボックス解を出すのか、あるいは特徴量を作りこむのか

(すなわち、このコンペではブラックボックスモデルが出されてもいいということ)

現在HomeCreditの活動範囲は中国・インド・インドネシア・ベトナム・フィリピン・ロシア・カザフスタン・米国・チェコスロバキア
の9カ国である。

## Abstract

* 0.77 is not enough
* models used in the home credit group is both log.reg and much sophisticated ML models
* 既に多くのcreditを借りている人にお金を貸すことは企業としてはいいことであるが、顧客をover-debtにしてしまう可能性があるので避けたい(AUCとしてはそれが良いという結果が出ていたとしても)
* 100以上の特徴を加えてモニタリングするよりも、50くらいの特徴でだいたい同じくらいの精度を出せる方がいい
* 最終的にはより簡単なモデルを使うにしても一度は複雑なモデルを試す
* modelの再構築時には古いモデルを通ってきた母集団で学習を行うため、そこで弾かれていたような人が紛れ込む可能性がある。
* 実際のモデルでは、性別・結婚状態・子供の数など、使えない変数がある。
* その時点での情報を入れれば推論の精度は良くなるがそれは入れられない
* Home Credit Groupのデータサイエンスチームが用意しているモデルは多数あり、1.顧客が最適なローン額を選ぶのを助けるモデル 2.riskから利子率を算出するモデル? 3.データの収集戦略を最適化するモデル 4.ローンの提案をするモデル 5.大規模な詐欺行為を調査するモデル 6. 顧客のクレジットカード制限の増減を決めるモデル
* ローンの契約は国や地域によって異なると考えられる(法的要請や産業構造の違いなどから)。応募用フォームは確かに異なるが現在データサイエンスチームが国ごとに大きく異ならないようにしているところである
* 収入は自己申告かタイポ
* Annuityは応募者が申請時に登録する月ごとの支払額(balanceやpaymentでは違う場合も)
* DAYS_EMPLOYEDの最大値はpensionerかunemployed

## Questions and Discussions

### Arthur Paulino
> Are the top public scores(~0.77 right now) any better?

Kirill
> So far our model is stronger, as our model had to meet quite a lot of business limitations.We will hopefully have time to publish our best comparable model during the competition.

kecsap
> Is your model a black box type one, or do you use the industrywise applied logistic regression with smart variables? I am really curious whether you can use hardcore ML algorithm in a world with financial authorities, requirements to understand and monitor the model, and with managers keen to understand what is behind the scoring engine.

Kirill
> We do use both ML models and log.reg on production for loan approval.
The ML model have to of course meet some the extra requirements so we need to:
1. make the "black-box" into "grey-box" by using methods as LIME.
2. make sure some forbidden or not preferable variables or relationships are not in the model.
3. limit number of final variables used for monitoring purposes.
4. deal with the fact that we are observing only some censored population(only those who we approved by previous model) -dealing with this is a bit harder in Black-box model

> Whether we use log.reg or ML models highly depends on market, portfolio and data.Especially for smaller or messier data we stuck to good old log.reg.On some markets like US is quite hard from regulatory perspective to use ML, but luckily for us in most of our markets the regulators allow us to use any model so even the darkest black-box.
In general we develop ML model for every model that we create.To have some benchmark and also to utilize some info to the log.reg.

Aseem
> By "make sure some forbidden or not preferable variables or relationships are not in the model" do you do this to remove bias? Like if you discover that certain community is being excluded? What is not preferable?

> Could you possibly elaborate on "limit number of final variables used for monitoring purposes"?

> "deal with the fact that we are observing only some censored population (only those who we approved by previous model) - dealing with this is a bit harder in Black-box model" - In this when say approved by previous model do you use series of models or are perhaps referring to some business rules here?

> "Whether we use log.reg. or ML models highly depends on market" and "In general we develop ML model for every model that we create" - Here by ML models do you mean models that are harder to explain? What is the distinction between the term "model" and "ML model" here?

> Guess about the multiple models.Are multiple models referring to models from other companies or government agencies? Is it legally required to make decisions based off their models? Or are their predictions used because they have access to information that you don't have access to make models upon?

Kirill
> For production model there could be many reasons to not wanting some relationship in the model.For example befire crisis it is quite probable that people with more approved loans are better than those with less, but if the crisis would start this relationship would be opposite. On top of that giving better scores to applicants with more loans would not be responsible lending as it would over-debt the clients.So this is the relationship we would normally not want in the model even if it makes sense from AUC point of view.Other relationships or features we want to exclude for regulatory and ethical reasons.

> Well if we would not limit number of features we would have models with 100+ features as they perform better.But implementation and monitoring of 100+ is not something we want to do especially if models with 50 features can come quite close in performance.Implementing 100+ features in production pose serious threat of error.By error some features can change even when they are on production - this is why we have to monitor them which again for 100+ features might be complicated.

> In this case by "model" I meant Logistic Regression, regularized Logistic regression, Decision tree...By "ML models" I meant models like XGB and other variations of gradient boosting, Neural networks, ...So we always build some harder model that is black-box even if we finally implement some easier model.

> We rebuild our models quite often.Every time we rebuild a model we can do it only on approved clients by the old model(we have no target for rejected clients).When we implement new model we stop using the old one, but the new model can start approving people who previously would be rejected by the old model not because they are good but simply because the new model did not learn that they are bad because they were not in the dataset for the new model.So for example if your old model would see that the people with "characteristic A" are really bad it would reject almost all of them - only those with best other characteristic would have a chance to be approved.The next model you will build on clients approved by the old model.Thus you would have small amount of people with "characteristic A" and they would not seem so bad, but inreality they are you just can't observe it in the data because the old model took only the best ones with "characteristic A".

Aseem
> So how do you remove the relationships? I don't have a lot of experience in using models in production.Just for 1 field.What we do is to business rules to not use the model in certain scenarios.But excluding certain relationships within the model is a new concept for me.How do you do that? Or do you also use business rules to exclude those relationships? Or do you somehow penalize those relationships?

> Got the problem with 100+ variables.Some manual checks need to be made usually to fix those errors.

> Regarding not having a target for rejected clients, is that because if you had that the new model would also learn the mistakes of the old model? The datacollection strategy was the reason I asked in the other comment whether you solely use models for deciding to approve or reject the loans.How was the model seeded initially? Solely based the credit scores? And then as the loans were repaid you got the data for your future models?

Kirill
> Honestly for this we have no simple guide.It highly depends on what kind of model you have, how complex are your data and how complex is the relationship you don't want.Our approach is  to not put predictors to the model that are forbidden or are strong;u correlated with the forbidden predictors.Then we look if the unwanted relationship is there (by unwanted relationship I mean also business illogical relation with risk within
one variable for example that people with higher DPD are better).If it is there we solve it ad hoc.

> New model learning old mistakes is not an issue.Issue is if new model does not learn something because the old model skewed the data.Yes exactly - we first approve clients by model 1.1 that for new markets is created expertly.In the future we will build model 2.1. but at that time we have target variable only for clients who were approved by 1.1 those rejected by 1.1. did not receive the loan so we don't know how they would behave if would have accepted them.Abd the chain of models continue like that.For each new model we are risking that the old skewed our population too much and have to test for it.

### Iorenzo palloni

> I read rules, and I wondered if I could use the Competotion Data as well as for the competition, alson to present my results in the thesis of the application of the methods that I was interested in applying.

Kirill
> Hello Lorenzo, sorry for taking long to reply.Let's discuss some data utilization in private chat.The data downloaded from Kaggle can definitely only be used with accordance to Kaggle rules.But maybe we can figure something out.I already asked our PR and Legal departments for their opinion.

### Yanni Papadakis

> Unfortunately, this is not a great application for Machine learning.
1. A number of your variables are not allowed for credit decisions in many countries, e.g. gender, marital status, number of children.
2. Other variables are problematic in more subtle way.By looking at prior applications you are not really developing a new model, but you are polling other people's models and it is not clear how these work.How can you measure the impact of this variable in an machine learning model? It's not easy.And, you have to let people know too many applications are bad for them.If on the other hand, somehow, many applications are good for the unsuspecting applicant then your credit score is manipulable in reverse (to get a better score one just needs to send multiple applications)
3. Your data appears to be just a snapshot of credit performance (for some unknown time), not including economic variables relating to economy-wide orioensity for bankruptcy.Hence, you have no clue how estimated probability of default will vary in the future.It appears, also, your combine loans from 4 currency areas with significantly different average probability of default between them.This way it is difficult to even try a stress test analysis of your model.

> Machine Learning should stick to its knittingm look-alike black-box models when there is no science-based model to couch data on and when the cost of missestimation is not high.

Kirill
> Honestly speaking for us it is interesting to find any model performing better than ours.So far the Machine Learning models with some smart feature aggregation worked best for us so this is why we wanted to see what Kaggle community cando with our data

> 1. **This model is not to be implemented so please feel free to use any variables**.Our goal is to see different approaches not to have production model.
2. If I understand correctly you are talking about reject inference problem.That some data are available only for clients that were approved by other models(some models are our old models and  some are models of other financial institiution - for data in CB).Yes I agree this is a challenge.Sadly we cannot publish our old models and models of other institutions are obviously not available even to us.But I believe challenge is what makes the competition interesting! Att the same time it gives participants a chance to see what problems they have to deal when modelling on real financial data.
3. I agree with you that providing mode information would allow using macroeconomics and use some trends in between contracts.Sadly again this is not possible from many aspects.

### Aseem Bansal

> Do you rely solely on models to approve or reject the loans?

Kirill
> We have quite many more models then just the ones for approval/rejection we have models to
1. help clients select optimal loan amount
2. choose fair pricing for clients - based on their  predicted risk.Good clients receive better interest rate.In quite many countries we even provide 0 interest rate products for best clients
3. optimize collection strategy
4. offer loands to our exciting clients (propensity to buy)
5. to search for Fraudulent behavior on bigger scale
6. increase/decrease limits on client's credit cards...
I depends on exact situation and market how will the model be used.Model in this competition will not go to production (for many reasons we cannot use it as production model), but if it would it would most likely be used for reject/approval, 1. 2. and 3.

### aisaac

> I have a quick question, is the loan agreement provided by Home Credit have a specific rules for each country or it applies to all countries?

Kirill
> To be honest as an data scientist I do not know much our loan agreement varies from country to country as it is a Legal document.I would guess it does a lot thought as every country has different law requirements and different industry standards.I can tell you that required fiekds on application form vary a lot, but soon it should be unfied as we are working on reducing the application field to the minimum in all countries

### Santhosh

> Could you please help me in understanding below:
1. One customer has an AMT_INCOME_TOTAL as 117000000 is that a data entry error?
2. Is that normal to have AMT_CREDIT > AMT_GOODS_PRICE
3. Could you explain a bit more about AMT_ANNUITY.
4. What does positive & negative sign mean in DAYS_EMPLOYED

Kirill
> 1. the income is self proclaimed and not verified.Or the sales assistant made mistake in number of 0
2. it could happen, I know  ine case but would like to double check if there are othres.
3. It is the monthly payment that client will send us(interest rate included) this is info at application time as some clients can change thier annuity which you would see in the other tables(balance and payment tables)
4. this is basically "if employed then datediff(days, application_date, start_employment) else <>" so for events in the past it should be negative.But in the data all events related to contracts are past.So all should be negative.Only case when it is positive if person is not employed at the moment of application

Meyk
> 2. As far as I know it could happen because AMT_GOODS_PRICE is not crucial parameter.More important factor is if client is able to pay instalments.When income / risl/ other parameters are convincing enough the goods_price can be lower than credit -correct me if I am wrong!

> Moreover the estimated value of goods - especially in case of real estate - are lower than the real value (to minimize the risk of loss in Poland)

Kirill
> Getting back to point
2. AMT_CREDIT > AMT_GOODS_PRICE is caused by insurance as insurance price is included in AMT_CREDIT while it is not included in AMT_GOODS_PRICE.So the AMT_GOODS_PRICE is X but the clients also can purchase insurance so the AMT_CREDIT = X + insurance.Please note that there are more types of insurance and that the flag NFLAGS_INSURED_ON_APPROVAL that is in the data corresponds to one specific type. So there could be cases where NFLAGS_INSURED_ON_APPROVAL = 0 and at the same time AMT_CREDIT>AMT_GOODS_PRICE( this would be because other insurance that the flag does not see)

fatihozturk
> there are many rows DAYS_EMPLOYED == 365243 (which means unemployed), however; same rows have AMT_INCOME_TOTAL values as well.Alson their values are quite similar to those employed.How could this be possible?

David
> I noticed that most of those with DAYS_EMPLOYED == 365243 are pensioners, with a few unemployed.A positive number indicates that they will be employed some time in the future, which is not the case here.

### BhaskarChaudhary
> Need a small clarification in file bureau_balance.csv The description for feature MONTHS_BALANCE reads as : "Month of balance relative to application date (-1 means the freshest balance date)"

> But then alot of values for that column have 0 value.Doesn't that mean that 0 refers to the freshest balance date and not -1?

> Answering my own question - a similar item in POS_CASH_balance.csv ecplaint this as
"0 means the information at applcation - opten it will be the same as -1 as many banks are not updating the information to Credit Bureau regularly"
