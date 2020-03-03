#-*- coding: utf-8 -*-
"""
Created on Sat Feb  9 13:47:03 2019

@author: Chris
"""

''
import matplotlib.pyplot as plt


#2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.
#
 #  A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.
#
 #  B. What insights can you suggest from the graph?
#
 #  C. Based on the graph, is the behavior of any of the providers concerning? Explain.


###THIS IS A NON-FUNCTIONING EXAMPLE... I can't give the full answer, but wanted to help out.


#
#Pseudo code... 
#Aggregate Provider IDs and Provider Payment Amounts to get a total count for:
    #Unpaid claims
    #Paid claims

#Validate the counts by providers...

#You could even create labels for the providers to label them in the plot.


plotLabels = ['ProviderID1','ProviderIDN'] #etc


#Produce the scatterplot as the answer to 2a
fig, ax = plt.subplots()
ax.scatter(UNPAIDAGG, PAIDAGG)
ax.grid(linestyle='-', linewidth='0.75', color='red')

fig = plt.gcf()
fig.set_size_inches(25, 25)
plt.rcParams.update({'font.size': 28})

for i, txt in enumerate(plotLabels):
    ax.annotate(txt, (UNPAIDAGG[i], PAIDAGG[i]))

plt.tick_params(labelsize=35)
plt.xlabel('#of Unpaid claims', fontsize=35)

plt.ylabel('#of Paid claims', fontsize=35)

plt.title('Scatterplot of Unpaid and Paid claims by Provider', fontsize=45)
plt.savefig('Paid_Unpaid_Scatterplot.png')