## Methodology
For the analysis, I identified two problems of interest: estimating the effect of individual policy interventions on case rates in 1 to 2 weeks, and evaluating the effect of dynamic response strategies. The second problem was not addressed here but is outlined below for contrast. 

### Part 1: Analyzing Policy Effectiveness
The first estimates the effect of individual policy interventions (e.g. containment, social distancing, etc.) on case rates in 1 to 2 weeks, at the national level as well as state level. Estimating the effectiveness of policies can help inform decision makers as well as understand policies with respect to local characteristics. 

To analyze individual policies' effectiveness, I use model-based methods based on outcome regression and propensity score weighting to estimate average treatment effect, identified using backdoor adjustment, on outcomes (e.g. case rate) 1 to 2 weeks out. 

Policies are differentiated so we model changes in policy strengths, while keeping non-differentiated values as confounders. Additional features such as lagged outcome values and temporal features are added to decorrelate the rows, in addition to confounding variables. By adjusting for history appropriately, we are able to learn treatment effect under time varying confounding and treatment confounder feedback. 

### Part 2: Analyzing Response Strategy
The second estimates the effect of dynamic response strategies at the state level with respect to (weaker) strategies enforced at the national level. Estimating the effectiveness of longitudinal response strategies can help us analyze government response as a whole. Comparing to the national level strategy gives us a realistic baseline to compare to (instead of “never intervene” control strategies that are often done for illustrative purposes). 

To analyze the dynamic response strategy, we can learn the conditional distributions of covariates and outcomes using regression models, then simulate covariates and outcomes under national and state level strategies according to the g-formula. 

Simulating individual states under national and state level strategies for slowing disease spread and rolling back on containment measures allows us to 1) measure how much more (or less) effective the state level strategy was over the national level strategy, and 2) compare strategies across states by looking at outcome trajectories relative to that simulated under the national level strategy. 


