ObjectPath
==========

[![Downloads](https://img.shields.io/pypi/dm/objectpath.svg)](https://pypi.python.org/pypi/objectpath/)
<!--[![License](https://img.shields.io/pypi/l/objectpath.svg)](https://pypi.python.org/pypi/objectpath/)-->
[![Build Status](https://travis-ci.org/adriank/ObjectPath.svg?branch=master)](https://travis-ci.org/adriank/ObjectPath)
[![Code Health](https://landscape.io/github/adriank/ObjectPath/master/landscape.png)](https://landscape.io/github/adriank/ObjectPath/master)
[![Coverage Status](https://coveralls.io/repos/adriank/ObjectPath/badge.png?branch=master)](https://coveralls.io/r/adriank/ObjectPath?branch=master)

The agile NoSQL query language for semi-structured data
-----------------------------------------------

**#Python #NoSQL #Javascript #JSON #nested-array-object**

ObjectPath is a query language similar to XPath or JSONPath, but much more powerful thanks to embedded arithmetic calculations, comparison mechanisms and built-in functions. This makes the language more like SQL in terms of expressiveness, but it works over JSON documents rather than relations. ObjectPath can be considered a full-featured expression language. Besides selector mechanism there is also boolean logic, type system and string concatenation available. On top of that, the language implementations (Python at the moment; Javascript is in beta!) are secure and relatively fast.

More at [ObjectPath site](http://objectpath.org/)

![ObjectPath img](http://adriank.github.io/ObjectPath/img/op-colors.png)

ObjectPath makes it easy to find data in big nested JSON documents. It borrows the best parts from E4X, JSONPath, XPath and SQL. ObjectPath is to JSON documents what XPath is to XML. Other examples to ilustrate this kind of relationship are:

| Scope  | Language |
|---|---|
| text documents  | regular expression  |
| XML  | XPath  |
| HTML  | CSS selectors  |
| JSON documents | ObjectPath |

Documentation
-------------

[ObjectPath Reference](http://objectpath.org/reference.html)
`````sh
$ git clone https://github.com/saithalavi/ObjectPath
$ cd ObjectPath
$ python shell.py file.json
`````

Python usage
----------------

`````sh
$ git clone https://github.com/saithalavi/ObjectPath
$ cd ObjectPath
$ python3
>>> data = {'user': [{'name': 'Ambu', 'age': '7.5'}, {'name': 'Amaan', 'age': 3.5}]}
>>> import objectpath
>>> tree = objectpath.Tree(data)
>>> tree.data
{'user': [{'name': 'Ambu', 'age': '7.5'}, {'name': 'Amaan', 'age': 3.5}]}
>>> """ tree.data returns a copy. So user cannot corrupt data handled by tree """
>>> tdata = tree.data
>>> tdata['user'][1]
{'name': 'Amaan', 'age': 3.5}
>>> del tdata['user'][1]
>>> tdata
{'user': [{'name': 'Ambu', 'age': '7.5'}]}
>>> del tdata['user'][0]
>>> tdata
{'user': []}
>>> tree.data
{'user': [{'name': 'Ambu', 'age': '7.5'}, {'name': 'Amaan', 'age': 3.5}]}
>>> tree.data
{'user': [{'name': 'Ambu', 'age': '7.5'}, {'name': 'Amaan', 'age': 4}]}
>>> tree.update("$[@.name is not 'Amaan'].age", 7)
>>> tree.update("$.user[@.name is not 'Amaan'].age", 7)
True
>>> tree.data
{'user': [{'name': 'Ambu', 'age': 7}, {'name': 'Amaan', 'age': 4}]}
>>> tree.update("$.user[@.name is not 'Amaan'].name", "Amjad")
True
>>> tree.data
{'user': [{'name': 'Amjad', 'age': 7}, {'name': 'Amaan', 'age': 4}]}
>>> tree.delete("$.user[@.name is not 'Amaan']")
True
>>> tree.data
{'user': [{'name': 'Amaan', 'age': 4}]}
>>> tree.delete("$.user[0]")
True
>>> tree.data
{'user': []}
>>> tree.delete("$")
True
>>> tree.data
{}
>>> data = [{'name': 'Amjad', 'age': 8, 'grades': ['A', 'A+']}, {'name': 'ami', 'age': 5}]
>>> tree = objectpath.Tree(data)
>>> tree.delete("$[0].grades")
True
>>> tree.data
[{'name': 'Amjad', 'age': 8}, {'name': 'ami', 'age': 5}]
>>> tree.update("$[2]", {'name': 'new', 'age': 5})
True
>>> tree.data
[{'name': 'Amjad', 'age': 8}, {'name': 'ami', 'age': 5}, {'name': 'new', 'age': 5}]
`````

Contributing & bugs
-------------------

I appreciate all contributions and bugfix requests for ObjectPath, however since I don't code in Python anymore, this library is not maintained as of now. Since I can't fully assure that code contributed by others meets quality standards, I can't accept PRs.

If you feel you could maintain this code, ping me. I'd be more than happy to transfer this repo to a dedicated ObjectPath organization on GitHub and give the ownership to someone with more time for this project than me.

License
-------

**MIT**
