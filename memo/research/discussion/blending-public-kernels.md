### blending public kernels (abortive attempt at "safe harbor" thread)

>This thread has ended up focusing on a specific issue, so I'm changing the title (originally "Random ideas: 'safe haven harbor' thread") as Tilli suggested in the comments.I'm restarting the original thread elsewhere, but it may die a natural death.

>I'll just boast about my superior overfitting ability: I've done literally nothing in this competition yet except combining public results (using weighted rank averages), and I've still managed to make it to top 2% on the public LB...for now.Goes to show how well you can overfit when you crowdsource the base models.I may discuss this in more detail later in the thread.

> The 6 models that make up my ensemble:

* Dromosys' version of the popular LightGBM model (not sure who made the first version, maybe olivier, but it has gone through many revisions by many people, some of which you can see by following the "forked from" links).
* kxx's tudyxgb model
* Scirpus's hybrid jeepy model
* James Shepherd's Dart model
* Prashant Kikani's stacked RF/ET/XGB/CB model
* David Salaza's PyTorch model

> Footnotes to the previous comment:

> 1. The weight are very lopsided (Unlike with Porto Seguro, equal weights do not work well at akk, at least with the current set of availablle kernels.)
2. You can probably reproduce my results with a little bit of effort, but that's also probably not the best use of your time(and submission quota) if your objective is to do well in this competition.The pyblic test set is small and (presumably) imbalanced, so results derived from fitting it are not likely to generalize well


Max
> Are you using rank for ensemble? Different models has different means of TARGET in submission files and it needs some scale transformation

Andy
> Yes ( as stated in the original post).As per common practice, I rescale the ranks to be in the 0 to 1 range by dividing by the number of cases.It's possible there are better ways to scale these results for blending, but in the Porto Seguro competition, ranks seemed to work best, so that's what I started with here.

### Mishunyayev
> I correctly understood that you used blending or stacking 6 models picking weights and nothing else?

Andy
> Yes I am using a blend of 6 models (using ranks) and nothing else.I didn't even write any code yet, unless you count the Excel RANK() function.I just downloaded the submission files and copied them into a spread sheet.Aside from converting the predictions to ranks and rescaling to be in the 0 to 1 range, all I did was introduce the weights and take an average.

Mishunyayev
> What are the ranks?

Andy
>The predictions are expressed as probabilities, and you can rank the rows from lowest to highest probability (assigning 1, 2, 3 etc. which is typically donw by a RANK() function, as in Excel or Pandas), then devide by the number of rows to make the result look like a probability so it will be accepted by the scorer.In this competition the scoring criterion depends only on the ranksm not on the specific values of the predictions, so if you do this for a single submission, it will not affect the score.That makes it useful as a concenient way to blend results that may have different scaling.

### Shahebaz
> Can someone briefly explain how does this lead to overfitting?

Andy
> OK, let me try this in stages:
1. You can't fit to the public test data directly, but you can fit it indirectly by making random changes in your predictions and repeatedly choosing the submissions with the best scoresAnd you could speed up this process by numerically estimating the gradient(and even the Hessian) of these scores and following the descending gradient.Provided the public test data are not too large, the final result would score very well on the public LB.But it would not generalize wekk, and it would score badly on the private LB.That's the simplest stage of overfitting.
2. In practice, you probably won't do that, but you might make random changes in your model's hyperparameters and follow a similar process.This will generalize somewhat better, because some ofthe changes you make are really improving themodel.But some of the changes are really not improving the model, just improving the fit to the public test data.If you pursue this process aggressively enough, your model woll almost certainly be overfit to public test data.However, there's a limit to how overfit it can get, since you're only allowed a limited number of submissions
3. Now suppose you come up with a whole bunch of different models, choose the ones with the best public LB scores, and combine the results.This is really a special case of #2 where which models to use is the hyperparameter that you're optimizing.Again, if you pursue this process aggressively enough, your model will end up overfit to the public test data, but again, there is a limit to your overfittingm because you have a limited number of submissions
4. 
