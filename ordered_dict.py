#https://www.geeksforgeeks.org/ordereddict-in-python/

'''
An OrderedDict is a dictionary subclass that remembers the order that keys were
first inserted. The only difference between dict() and OrderedDict() is that:

OrderedDict preserves the order in which the keys are inserted. A regular dict
doesn’t track the insertion order, and iterating it gives the values in an
arbitrary order. By contrast, the order the items are inserted is remembered
by OrderedDict.

Attention:
Until recently, Python dictionaries did not preserve the order in which items
were added to them. For instance, you might type {'fruits': ['apples', 'oranges'],
'vegetables': ['carrots', 'peas']} and get back {'vegetables': ['carrots', 'peas'],
 'fruits': ['apples', 'oranges']}. If you wanted a dictionary that preserved order,
 you could use the OrderedDict class in the standard library module collections.
However, this situation is changing. Standard dict objects preserve order in the
reference (CPython) implementations of Python 3.5 and 3.6, and this order-preserving
property is becoming a language feature in Python 3.7.

You might think that this change makes the OrderedDict class obsolete.
However, there are at least two good reasons to continue using OrderedDict.
First, relying on standard dict objects to preserve order will cause your code t
o break on versions of CPython earlier than 3.5 and on some alternative
implementations of Python 3.5 and 3.6. Second, using an OrderedDict
communicates your intention to rely on the order of items in your dictionary
being preserved, both to human readers of your code and to the third-party
libraries you call within it.

'''

#OrderedDict demonstration
from collections import OrderedDict

print('"This is a Dict:\n"')
d = {}
d['a'] = 1
d['b'] = 2
d['c'] = 3
d['d'] = 4

for key, value in d.items():
    print(key, value)


print("\nThis is an Ordered Dict:\n")
od = OrderedDict()
od['a'] = 1
od['b'] = 2
od['c'] = 3
od['d'] = 4

for key, value in od.items():
    print(key, value)
