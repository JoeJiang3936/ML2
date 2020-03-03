#-*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:31:47 2019

@author: Chris
"""
#import libraries
import numpy as np
import numpy.lib.recfunctions as rfn  #used to manipulate structured arrays
from collections import OrderedDict  #dictionary subclass that remembers the

#order that the keys were first inserted

#NumPy Cheatsheet - https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf


##HW notes:
"""
A medical claim is denoted by a claim number ('Claim.Number'). Each claim consists of one or more medical lines denoted by a claim line number ('Claim.Line.Number').

1. J-codes are procedure codes that start with the letter 'J'.

     A. Find the number of claim lines that have J-codes.

     B. How much was paid for J-codes to providers for 'in network' claims?

     C. What are the top five J-codes based on the payment to providers?



2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.

    A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.

    B. What insights can you suggest from the graph?

    C. Based on the graph, is the behavior of any of the providers concerning? Explain.



3. Consider all claim lines with a J-code.

     A. What percentage of J-code claim lines were unpaid?

     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

     C. How accurate is your model at predicting unpaid claims?

      D. What data attributes are predominately influencing the rate of non-payment?
"""


#Read the two first two lines of the file.
with open("claim.sample.csv", "r") as f:
    print(f.readline())
    print(f.readline())


#Colunn names that will be used in the below function, np.genfromtxt
names = [
    "V1",
    "Claim.Number",
    "Claim.Line.Number",
    "Member.ID",
    "Provider.ID",
    "Line.Of.Business.ID",
    "Revenue.Code",
    "Service.Code",
    "Place.Of.Service.Code",
    "Procedure.Code",
    "Diagnosis.Code",
    "Claim.Charge.Amount",
    "Denial.Reason.Code",
    "Price.Index",
    "In.Out.Of.Network",
    "Reference.Index",
    "Pricing.Index",
    "Capitation.Index",
    "Subscriber.Payment.Amount",
    "Provider.Payment.Amount",
    "provider.Index",
    "Subscriber.Index",
    "Subprovider.Index",
    "Claim.Type",
    "Claim.Subscriber.Type",
    "Claim.Pre.Prince.Index",
    "Claim.Current.Status",
    "Network.ID",
    "Agreement.ID",
]


#https://docs.scipy.org/doc/numpy-1.12.0/reference/arrays.dtypes.html
#These are the data types or dtypes that will be used in the below function, np.genfromtxt()
types = [
    "S8",
    "f8",
    "i4",
    "i4",
    "S14",
    "S6",
    "S6",
    "S6",
    "S4",
    "S9",
    "S7",
    "f8",
    "S5",
    "S3",
    "S3",
    "S3",
    "S3",
    "S3",
    "f8",
    "f8",
    "i4",
    "i4",
    "i4",
    "S3",
    "S3",
    "S3",
    "S4",
    "S14",
    "S14",
]


#NumPy Structured Arrays: https://docs.scipy.org/doc/numpy/user/basics.rec.html
#Though... I like this Structured Array explanation better in some cases: https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html
#np.genfromtxt:  https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html

#read in the claims data into a structured numpy array
CLAIMS = np.genfromtxt(
    "claim.sample.csv",
    dtype=types,
    delimiter=",",
    names=True,
    usecols=[
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
    ],
)

#print dtypes and field names
print(CLAIMS.dtype)

#Notice the shape differs since we're using structured arrays.
print(CLAIMS.shape)

#However, you can still subset it to get a specific row.
print(CLAIMS[0])

#Subset it to get a specific value.
print(CLAIMS[0][1])

#Get the names
print(CLAIMS.dtype.names)

#Subset into a column
print(CLAIMS["MemberID"])

#Subset into a column and a row value
print(CLAIMS[0]["MemberID"])


#String Operations in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.char.html

#Sorting, Searching, and Counting in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.sort.html

#You might see issues here: https://stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl

#If you do, encode as a unicode byte object
#A test string
test = "J"
test = test.encode()

#A test NumPy array of type string
testStrArray = np.array(["Ja", "JA", "naJ", "na"], dtype="S9")

#Showing what the original string array looks like
print("Original String Array: ", testStrArray)

#Now try using startswith()
Test1Indexes = np.core.defchararray.startswith(testStrArray, test, start=0, end=None)
testResult1 = testStrArray[Test1Indexes]

#Showing what the original subset string array looks like with startswith()
print("Subset String Array with startswith(): ", testResult1)

#Now try using find()
TestIndexes = np.flatnonzero(np.core.defchararray.find(testStrArray, test) != -1)

testResult2 = testStrArray[TestIndexes]

#Showing what the original subset string array looks like with find()
print("Subset String Array with find(): ", testResult2)

#Try startswith() on CLAIMS
JcodeIndexes = np.flatnonzero(
    np.core.defchararray.startswith(CLAIMS["ProcedureCode"], test, start=1, end=2) != -1
)

np.set_printoptions(threshold=500, suppress=True)

#Using those indexes, subset CLAIMS to only Jcodes
Jcodes = CLAIMS[JcodeIndexes]

print(Jcodes)


#Try find() on CLAIMS
JcodeIndexes = np.flatnonzero(
    np.core.defchararray.find(CLAIMS["ProcedureCode"], test, start=1, end=2) != -1
)

#Using those indexes, subset CLAIMS to only Jcodes
Jcodes = CLAIMS[JcodeIndexes]

print(Jcodes)

print(Jcodes.dtype.names)

#QUESTION: How do you find the number of claim lines that have J-codes with "Jcodes"?
#You can figure this out. :)


#QUESTION: How much was paid for J-codes to providers for 'in network' claims?
#Give this a try on your own after viewing the example code below.


#Sorted Jcodes, by ProviderPaymentAmount
Sorted_Jcodes = np.sort(Jcodes, order="ProviderPaymentAmount")


#Reverse the sorted Jcodes (A.K.A. in descending order)
Sorted_Jcodes = Sorted_Jcodes[::-1]
#[7, 6, 5, 4, 3, 2, 1]

#What are the top five J-codes based on the payment to providers?

#We still need to provider the data
print(Sorted_Jcodes[:10])

#You can subset it...
ProviderPayments = Sorted_Jcodes["ProviderPaymentAmount"]
Jcodes = Sorted_Jcodes["ProcedureCode"]

#recall their data types
Jcodes.dtype
ProviderPayments.dtype

#get the first three values for Jcodes
Jcodes[:3]

#get the first three values for ProviderPayments
ProviderPayments[:3]

#Join arrays together
arrays = [Jcodes, ProviderPayments]

#https://www.numpy.org/devdocs/user/basics.rec.html
Jcodes_with_ProviderPayments = rfn.merge_arrays(arrays, flatten=True, usemask=False)

#What does the result look like?
print(Jcodes_with_ProviderPayments[:3])

Jcodes_with_ProviderPayments.shape

#providerBy JCodes using a dictionary
JCode_dict = {}

#Aggregate with Jcodes - code  modifiedfrom a former student's code, Anthony Schrams
for aJCode in Jcodes_with_ProviderPayments:
    if aJCode[0] in JCode_dict.keys():
        JCode_dict[aJCode[0]] += aJCode[1]
    else:
        aJCode[0] not in JCode_dict.keys()
        JCode_dict[aJCode[0]] = aJCode[1]

#sum the JCodes
np.sum([v1 for k1, v1 in JCode_dict.items()])

#create an OrderedDict (which we imported from collections): https://docs.python.org/3.7/library/collections.html#collections.OrderedDict
#Then, sort in descending order
JCodes_PaymentsAgg_descending = OrderedDict(
    sorted(JCode_dict.items(), key=lambda aJCode: aJCode[1], reverse=True)
)

#print the results
print(JCodes_PaymentsAgg_descending)
