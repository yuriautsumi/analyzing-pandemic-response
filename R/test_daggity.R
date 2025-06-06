install.packages('dagitty')
library(dagitty)

q = dagitty('dag {

  Z -> A -> Y

  Z <- V -> Y

  Z <- W -> A}')

plot(graphLayout(q))

adjustmentSets(q, 'A', 'Y', type = 'all')
