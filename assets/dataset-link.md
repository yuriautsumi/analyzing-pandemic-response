## Data

Data from the following sources were combined for this study:

1. Oxford Covid-19 Government Response Tracker (OxCGRT) [[Source](https://www.bsg.ox.ac.uk/research/covid-19-government-response-tracker)]
2. U.S. Census Bureau [[Source](https://data.census.gov)]
3. Hospital Admissions [[Source](https://covid.cdc.gov/covid-data-tracker/#trends_weeklyhospitaladmissions_select_00)]
4. Hospital Coverage [[Source](https://healthdata.gov/Hospital/COVID-19-Hospital-Data-Coverage-Report/v4wn-auj8/about_data)
5. Google Mobility [[Source](https://www.google.com/covid19/mobility/data_documentation.html?hl=en)]
6. Vaccination Rate [[Source](https://ourworldindata.org/us-states-vaccinations)]
7. Miscellaneous: Political Leaning [[Source](https://en.wikipedia.org/wiki/2020_United_States_presidential_election#cite_note-FEC-2)], Population Density [[Source](https://wernerantweiler.ca/blog.php?item=2020-04-12)], Pandemic Wave [[Source 1](https://www.pewresearch.org/politics/2022/03/03/the-changing-political-geography-of-covid-19-over-the-last-two-years/), [Source 2](https://pubmed.ncbi.nlm.nih.gov/38437606/)]


### Policy data

The OxCGRT data provides the primary data on non-pharmaceutical interventions enforced at state and national levels during the COVID-19 pandemic (early 2020-2022). Categories include: containment, health, economic, and vaccine interventions. 

Although the raw data encodes the policy level that is in effect (integer value indicates strength of policy), I re-encoded “intervention” policies as the change in policy value in the last week, similar to [Barros et al. (2022)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9518862/), since I am interested in the effect of policy *intervention*. The full vector of policies in place are treated as confounders. 


### Outcomes

I consider the policies' effect on case rates per 100k people in a specified forecast window (e.g. 0 days, 7 days, 14 days). 

Past studies also considered the effective reproduction number ($R_t$) of the disease, the average number of new infections caused by an infected individual in the susceptible population ([Flaxman et al. 2020](https://www.nature.com/articles/s41586-020-2405-7), [Barros et al. 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9518862/), [Sun et al. 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8651490/)). However, this metric is not directly observed and must often be estimated under assumptions about the infection dynamics or from other estimated parameters, which leads to uncertainty ([Delamater et al. 2019](https://wwwnc.cdc.gov/eid/article/25/1/17-1901_article)). I used a more straight forward outcome (case rate per 100k people); similar outcomes have been modeled in previous literature ([Ma et al. 2021](http://arxiv.org/abs/2106.01315)). 

From a public health standpoint, however, it may be more informative to model outcomes that help us more explicitly analyze shape of the infection curve (e.g. flatness), since the goal is often to flatten the curve to avoid resource shortage and reduce disease severity to slow epidemic spread ([Williams 2020](https://newsnetwork.mayoclinic.org/discussion/covid-19-flattening-the-curve/)). Aside from reproduction number, previous studies have also looked at survival time until case rates exceed some threshold ([Baird et al., 2024](https://pubmed.ncbi.nlm.nih.gov/38437606/)). 

### Confounders

To estimate the effect of policy on outcome, it is also important to adjust for confounding variables. When estimating the effect of intervention ($A$) on outcome ($Y$), confounding variables are static (baseline) or time-varying variables ($L$) that have an arrow to interventions ($A$) and outcomes ($Y$) in the causal graph where $L\rightarrow A$, $L\rightarrow Y$, $A\rightarrow Y$. Controlling for confounders ($L$) allows us to block backdoor paths from treatment to outcome, which allows us to isolate effect of treatment on outcome through frontdoor paths (i.e. paths with arrows that emnate from treatment). In layman terms, controlling for confounders helps avoid over or under-inflating the effect of treatment on outcome. 

A few confounders I included are the following:
* Demographic factors (socioeconomic indicators, age distributions, political leanings, population density)
* Clinical care factors (clinical care availability, hospital admissions per week)
* Vaccination rates (percentage of partially vaccinated, fully vaccinated)
* Temporal factors (pandemic wave, time of year)
* Lagged outcomes

Confounders were chosen from regression analyses to verify associations and from referencing previous literature ([Barros et al. 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9518862/), [Baird et al., 2024](https://pubmed.ncbi.nlm.nih.gov/38437606/)). Since the data is time-series, I also included temporal factors such as seasonal and pandemic wave indicators and lagged outcomes, to ensure that rows are decorrelated. To test that sufficient confounders have been controlled for, I conducted conditional independence (ignorability) tests for each pair of treatment and outcome, i.e. $A\perp Y|L$. 

### Mediators

For the mediation study, I looked at how mobility patterns and vaccination rates mediate the effect of containment and vaccine interventions on outcomes, respectively. 

For mobility patterns, I used the Google Mobility data, which records percent change in mobility patterns compared to a baseline in a 5 week window for various locations (e.g. residential, workplace, grocery & pharmacy). I removed residential patterns since its patterns were less variable (people spend a nontrivial amount of time at home, regardless of a pandemic) and park patterns since it was very correlated with seasonality (peaks in summer, plunges in winter). I included data from the following location categories: transit, retail & recreation, grocery & pharmacy, and workplaces. 

For vaccine interventions, I used data on percentage of vaccinated and fully vaccinated people per state. 

