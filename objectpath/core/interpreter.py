#!/usr/bin/env python

# This file is part of ObjectPath released under MIT license.
# Copyright (C) 2010-2014 Adrian Kalbarczyk

import copy
import sys, re
from .parser import parse
from objectpath.core import *
import objectpath.utils.colorify as color  # pylint: disable=W0614
from objectpath.utils import flatten, filter_dict, timeutils, skip
from objectpath.utils.json_ext import py2JSON
from objectpath.core import ITER_TYPES, generator, chain
from objectpath.utils.debugger import Debugger

EPSILON = 0.0000000000000001  #this is used in float comparison
RE_TYPE = type(re.compile(''))
# setting external modules to 0, thus enabling lazy loading. 0 ensures that Pythonic types are never matched.
# this way is efficient because if statement is fast and once loaded these variables are pointing to libraries.
ObjectId = generateID = calendar = escape = escapeDict = unescape = unescapeDict = 0


def create_path_from_tree(tree):
  if isinstance(tree, tuple) and tree:
      op = tree[0]
      if op == "[":
        if len(tree) == 1:
          return "[]"
        elif len(tree) == 2:
          if isinstance(tree[1], list):
              return "[{}]".format(", ".join([str(create_path_from_tree(x)) for x in tree[1]]))
          else:
              return "[{}]".format(str(create_path_from_tree(tree[1])))
        else:
          return "{}[{}]".format(str(create_path_from_tree(tree[1])), str(create_path_from_tree(tree[2])))

      if len(tree) > 2:
        if op in BINARY_OPS:
          return create_path_from_tree(tree[1]) + op + create_path_from_tree(tree[2])
        if op in BINARY_OPS_WITH_SPACE:
          return create_path_from_tree(tree[1]) + " " + op + " " + create_path_from_tree(tree[2])
        else:
          raise InvalidPath("create_path_from_tree: Unsupported BINARY_OP: {}".format(op))

      if op == "(root)":
        return "$"
      elif op == "name":
        return tree[1]
      elif op == "(current)":
        return "@"
      elif op in ["-", "+"]:
        return op + create_path_from_tree(tree[1])
      elif op in ["not"]:
        return op + " " + create_path_from_tree(tree[1])
  else:
      if isinstance(tree, str):
        return "'" + tree + "'"
      return str(tree)
  raise InvalidPath("create_path_from_tree: Unsupported tree: {}".format(str(tree)))


class Tree(Debugger):
  _REGISTERED_FUNCTIONS = {}

  @classmethod
  def register_function(cls, name, func):
    """
		This method is used to add custom functions not catered for by default
		:param str name: The name by which the function will be referred to in the expression
		:param callable func: The function
		:return:
		"""
    cls._REGISTERED_FUNCTIONS[name] = func

  def __init__(self, obj, cfg=None):
    self._locked = False
    self._data = None
    if not cfg:
      cfg = {}
    self.D = cfg.get("debug", False)
    self.setObjectGetter(cfg.get("object_getter", None))
    self.setData(obj)
    self.current = self.node = None
    self._expr_cache = {}
    if self.D: super(Tree, self).__init__()
    self._locked = True

  @property
  def data(self):
    if not self._locked:
      return self._data
    return copy.deepcopy(self._data)

  @data.setter
  def data(self, newdata):
    if type(newdata) in ITER_TYPES + [dict]:
      self._data = copy.deepcopy(newdata)
      self.sanitize_tree_state()
    else:
      raise ValueError("Type: {} is not supported".format(type(newdata)))

  def getData(self):
    return self.data

  def sanitize_tree_state(self):
    self._expr_cache = {}

  def setData(self, obj):
    self.data = obj

  def setObjectGetter(self, object_getter_cb):
    if callable(object_getter_cb):
      self.object_getter = object_getter_cb
    else:

      def default_getter(obj, attr):
        try:
          return obj.__getattribute__(attr)
        except AttributeError:
          if self.D:
            self.end(color.op(".") + " returning '%s'", color.bold(obj))
          return obj

      self.object_getter = default_getter

  def compile(self, expr):

    def replace_dummy_path(tree):
      if tree == ('.', ('(root)', 'rs'), ('name', 'dummy')):
        return tuple(('(root)', 'rs'))
      elif isinstance(tree, tuple):
        return tuple((replace_dummy_path(i) for i in tree))
      else:
        return tree

    path = expr.strip()

    if path in self._expr_cache:
      return self._expr_cache[path]

    if isinstance(self.data, list):
      if path == "$":
        return ('(root)', 'rs')
      if path.startswith("$"):
        suffix = path[1:].strip()
        if suffix.startswith("["):
            """ The parse seems to fail when root is a list """
            newpath = "$.dummy" + suffix
            tree = self._expr_cache[expr] = replace_dummy_path(parse(newpath, self.D))
            return tree
    ret = self._expr_cache[expr] = parse(expr, self.D)
    return ret

  def execute(self, expr):
    D = self.D
    if D: self.start("Tree.execute")

    if type(expr) == str and expr.strip() == "$":
      return self.data

    TYPES = [str, int, float, bool, generator, chain]
    try:
      TYPES += [long]
    except NameError:
      pass

    # TODO change to yield?
    def exe(node):
      """
				node[0] - operator name
				node[1:] - params
			"""
      types = [
          str, timeutils.datetime.time, timeutils.datetime.date,
          timeutils.datetime.datetime
      ]
      try:
        types += [unicode]
      except:
        pass
      if D: self.start("executing node %s", color.bold(self.cleanOutput(node)))
      type_node = type(node)
      if node is None or type_node in TYPES:
        return node
      elif type_node in types:
        return node
      elif type_node is list:
        return (exe(n) for n in node)
      elif type_node is dict:
        ret = {}
        for i in node.items():
          ret[exe(i[0])] = exe(i[1])
        return ret
      op = node[0]
      if op == "or":
        if D: self.debug("%s or %s", node[1], node[2])
        return exe(node[1]) or exe(node[2])
      elif op == "and":
        if D: self.debug("%s and %s", node[1], node[2])
        return exe(node[1]) and exe(node[2])
      elif op == "+":
        if len(node) > 2:
          fst = exe(node[1])
          snd = exe(node[2])
          if None in (fst, snd):
            return fst or snd
          typefst = type(fst)
          typesnd = type(snd)
          if typefst is dict:
            try:
              fst.update(snd)
            except Exception:
              if type(snd) is not dict:
                raise ProgrammingError(
                    "Can't add value of type %s to %s" % (
                        color.bold(
                            PY_TYPES_MAP.
                            get(type(snd).__name__,
                                type(snd).__name__)
                        ), color.bold("object")
                    )
                )
            return fst
          if typefst is list and typesnd is list:
            if D: self.debug("both sides are lists, returning '%s'", fst + snd)
            return fst + snd
          if typefst in ITER_TYPES or typesnd in ITER_TYPES:
            if typefst not in ITER_TYPES:
              fst = [fst]
            elif typesnd not in ITER_TYPES:
              snd = [snd]
            if D: self.debug("at least one side is a generator and the other is an iterable, returning chain")
            return chain(fst, snd)
          if typefst in NUM_TYPES:
            try:
              return fst + snd
            except Exception:
              return fst + float(snd)
          if typefst in STR_TYPES or typesnd in STR_TYPES:
            if D: self.info("doing string comparison '%s' is '%s'", fst, snd)
            if sys.version_info[0] < 3:
              if typefst is unicode:
                fst = fst.encode("utf-8")
              if typesnd is unicode:
                snd = snd.encode("utf-8")
            return str(fst) + str(snd)
          try:
            timeType = timeutils.datetime.time
            if typefst is timeType and typesnd is timeType:
              return timeutils.addTimes(fst, snd)
          except Exception:
            pass
          if D: self.debug("standard addition, returning '%s'", fst + snd)
          return fst + snd
        else:
          return exe(node[1])
      elif op == "-":
        if len(node) > 2:
          fst = exe(node[1])
          snd = exe(node[2])
          try:
            return fst - snd
          except Exception:
            typefst = type(fst)
            typesnd = type(snd)
            timeType = timeutils.datetime.time
            if typefst is timeType and typesnd is timeType:
              return timeutils.subTimes(fst, snd)
        else:
          return -exe(node[1])
      elif op == "*":
        return exe(node[1])*exe(node[2])
      elif op == "%":
        return exe(node[1]) % exe(node[2])
      elif op == "/":
        return exe(node[1])/float(exe(node[2]))
      elif op == ">":
        if D: self.debug("%s > %s, %s", node[1], node[2], node[1] > node[2])
        return exe(node[1]) > exe(node[2])
      elif op == "<":
        return exe(node[1]) < exe(node[2])
      elif op == ">=":
        return exe(node[1]) >= exe(node[2])
      elif op == "<=":
        return exe(node[1]) <= exe(node[2])
      # TODO this algorithm produces 3 for 1<2<3 and should be true
      # elif op in "<=>=":
      # 	fst=exe(node[1])
      # 	snd=exe(node[2])
      # 	if op==">":
      # 		return fst > snd and snd or False
      # 	elif op=="<":
      # 		return fst < snd and snd or False
      # 	elif op==">=":
      # 		return fst >= snd and snd or False
      # 	elif op=="<=":
      # 		return fst <= snd and snd or False
      elif op == "not":
        fst = exe(node[1])
        if D: self.debug("doing not '%s'", fst)
        return not fst
      elif op == "in":
        fst = exe(node[1])
        snd = exe(node[2])
        if D: self.debug("doing '%s' in '%s'", node[1], node[2])
        if type(fst) in ITER_TYPES and type(snd) in ITER_TYPES:
          return any(
              x in max(fst, snd, key=len) for x in min(fst, snd, key=len)
          )
        return exe(node[1]) in exe(node[2])
      elif op == "not in":
        fst = exe(node[1])
        snd = exe(node[2])
        if D: self.debug("doing '%s' not in '%s'", node[1], node[2])
        if type(fst) in ITER_TYPES and type(snd) in ITER_TYPES:
          return not any(
              x in max(fst, snd, key=len) for x in min(fst, snd, key=len)
          )
        return exe(node[1]) not in exe(node[2])
      elif op in ("is", "is not"):
        if D: self.debug("found operator '%s'", op)
        # try:
        fst = exe(node[1])
        # except Exception as e:
        # 	if D: self.debug("NOT ERROR! Can't execute node[1] '%s', error: '%s'. Falling back to orginal value.",node[1],str(e))
        # 	fst=node[1]
        # try:
        snd = exe(node[2])
        # except Exception as e:
        # 	if D: self.debug("NOT ERROR! Can't execute node[2] '%s', error: '%s'. Falling back to orginal value.",node[2],str(e))
        # 	snd=node[2]
        if op == "is" and fst == snd:
          return True

        # this doesn't work for 3 is not '3'
        # if op == "is not" and fst != snd:
        # 	return True
        typefst = type(fst)
        typesnd = type(snd)
        if D: self.debug("type fst: '%s', type snd: '%s'", typefst, typesnd)
        if typefst in STR_TYPES:
          if D: self.info("doing string comparison '\"%s\" is \"%s\"'", fst, snd)
          ret = str(fst) == str(snd)
        elif typefst is float or typesnd is float:
          if D: self.info("doing float comparison '%s is %s'", fst, snd)
          try:
            ret = abs(float(fst) - float(snd)) < EPSILON
          except:
            ret = False
        elif typefst is int or typesnd is int:
          if D: self.info("doing integer comparison '%s is %s'", fst, snd)
          try:
            ret = int(fst) == int(snd)
          except:
            ret = False
        elif typefst is list and typesnd is list:
          if D: self.info("doing array comparison '%s' is '%s'", fst, snd)
          ret = fst == snd
        elif typefst is dict and typesnd is dict:
          if D: self.info("doing object comparison '%s' is '%s'", fst, snd)
          ret = fst == snd
        elif fst is None or snd is None:
          if fst is None and snd is None:
            # this executes only for "is not"
            ret = True
          else:
            ret = (fst or snd) is None
            if D: self.info(
                "doing None comparison %s is %s = %s", color.bold(fst), color.bold(snd),
                color.bold(not not (fst or snd))
              )
        else:
          if D: self.info("can't compare %s and %s. Returning False", self.cleanOutput(fst), self.cleanOutput(snd))
          ret = False
        # else:
        # 	try:
        # 		global ObjectId
        # 		if not ObjectId:
        # 			from bson.objectid import ObjectId
        # 		if typefst is ObjectId or typesnd is ObjectId:
        # 			if D: self.info("doing MongoDB objectID comparison '%s' is '%s'",fst,snd)
        # 			ret=str(fst)==str(snd)
        # 		else:
        # 			if D: self.info("doing standard comparison '%s' is '%s'",fst,snd)
        # 			ret=fst is snd
        # 	except Exception:
        # 		pass
        if op == "is not":
          if D: self.info("'is not' found. Returning %s", not ret)
          return not ret
        else:
          if D: self.info("returning %s is %s => %s", color.bold(self.cleanOutput(fst)), color.bold(self.cleanOutput(snd)), color.bold(ret))
          return ret
      elif op == "re":
        return re.compile(exe(node[1]))
      elif op == "matches":
        fst = exe(node[1])
        snd = exe(node[2])
        if type(fst) not in STR_TYPES+[RE_TYPE]:
          raise Exception("operator " + color.bold("matches") + " expects regexp on the left. Example: 'a.*d' matches 'abcd'")
        if type(snd) in ITER_TYPES:
          for i in snd:
            if not not re.match(fst, i):
              return True
          return False
        else:
          # regex matches string
          return not not re.match(fst, snd)
      # elif op=="(literal)":
      # 	fstLetter=node[1][0]
      # 	if fstLetter is "'":
      # 		return node[1][1:-1]
      # 	elif fstLetter.isdigit:
      # 		return int(node[1])
      elif op == "(root)":  # this is $
        return self.data
      # elif op=="(node)":# this is !
      # 	if D: self.debug("returning node %s",self.node)
      # 	return self.node
      elif op == "(current)":  # this is @
        if D: self.debug("returning current node: \n  %s", color.bold(self.current))
        return self.current
      elif op == "name":
        return node[1]
      elif op == ".":
        fst = node[1]
        if type(fst) is tuple:
          fst = exe(fst)
        typefst = type(fst)
        if D: self.debug(color.op(".") + " left is '%s'", color.bold(self.cleanOutput(fst)))
        # try:
        if node[2][0] == "*":
          if D:
            self.end(
                color.op(".") + " returning '%s'",
                color.bold(typefst in ITER_TYPES and fst or [fst])
            )
          return fst  # typefst in ITER_TYPES and fst or [fst]
        # except:
        # 	pass
        snd = exe(node[2])
        if D: self.debug(color.op(".") + " right is '%s'", color.bold(snd))
        if typefst in ITER_TYPES:
          if D: self.debug(
                color.op(".") + " filtering %s by %s", color.bold(self.cleanOutput(fst)),
                color.bold(snd)
            )
          if type(snd) in ITER_TYPES:
            return filter_dict(fst, list(snd))
          else:
            # if D: self.debug(list(fst))
            return (e[snd] for e in fst if type(e) is dict and snd in e)
        try:
          if D: self.end(color.op(".") + " returning '%s'", fst.get(snd))
          return fst.get(snd)
        except Exception:
          if isinstance(fst, object):
            return self.object_getter(fst, snd)
          if D: self.end(color.op(".") + " returning '%s'", color.bold(fst))
          return fst
      elif op == "..":
        fst = flatten(exe(node[1]))
        if node[2][0] == "*":
          if D: self.debug(color.op("..") + " returning '%s'", color.bold(fst))
          return fst
        # reduce objects to selected attributes
        snd = exe(node[2])
        if D: self.debug(
              color.op("..") + " finding all %s in %s", color.bold(snd),
              color.bold(self.cleanOutput(fst))
          )
        if type(snd) in ITER_TYPES:
          ret = filter_dict(fst, list(snd))
          if D: self.debug(color.op("..") + " returning %s", color.bold(ret))
          return ret
        else:
          ret = chain.from_iterable(
              type(x) in ITER_TYPES and x or [x]
              for x in (e[snd] for e in fst if snd in e)
          )
          # print list(chain(*(type(x) in ITER_TYPES and x or [x] for x in (e[snd] for e in fst if snd in e))))
          if D: self.debug(color.op("..") + " returning %s", color.bold(self.cleanOutput(ret)))
          return ret
      elif op == "[":
        len_node = len(node)
        # TODO move it to tree generation phase
        if len_node is 1:  # empty list
          if D: self.debug("returning an empty list")
          return []
        if len_node is 2:  # list - preserved to catch possible event of leaving it as '[' operator
          if D: self.debug("doing list mapping")
          return [exe(x) for x in node[1]]
        if len_node is 3:  # selector used []
          fst = exe(node[1])
          # check against None
          if not fst:
            return fst
          selector = node[2]
          if D:
            self.debug(
              "\n  found selector '%s'.\n  executing on %s", color.bold(selector),
              color.bold(fst)
            )
          selectorIsTuple = type(selector) is tuple

          if selectorIsTuple and selector[0] is "[":
            nodeList = []
            nodeList_append = nodeList.append
            for i in fst:
              if D: self.debug("setting self.current to %s", color.bold(i))
              self.current = i
              nodeList_append(
                  exe((selector[0], exe(selector[1]), exe(selector[2])))
              )
            if D: self.debug(
                  "returning %s objects: %s", color.bold(len(nodeList)),
                  color.bold(nodeList)
              )
            return nodeList

          if selectorIsTuple and selector[0] == "(current)":
            if D:
              self.warning(
                  color.bold("$.*[@]") + " is eqivalent to " +
                  color.bold("$.*") + "!"
              )
            return fst

          if selectorIsTuple and selector[0] in SELECTOR_OPS:
            if D: self.debug(
                  "found %s operator in selector, %s", color.bold(selector[0]),
                  color.bold(selector)
              )
            if type(fst) is dict:
              fst = [fst]
            # TODO move it to tree building phase
            if type(selector[1]) is tuple and selector[1][0] == "name":
              selector = (selector[0], selector[1][1], selector[2])
            selector0 = selector[0]
            selector1 = selector[1]
            selector2 = selector[2]

            def exeSelector(fst):
              for i in fst:
                if D:
                  self.debug("setting self.current to %s", color.bold(i))
                  self.debug("  s0: %s\n  s1: %s\n  s2: %s\n  Current: %s", selector0, selector1, selector2, i)
                self.current = i
                if selector0 == "fn":
                  yield exe(selector)
                # elif type(selector1) in STR_TYPES and False:
                # 	if D: self.debug("found string %s", type(i))
                # 	try:
                # 		if exe((selector0,i[selector1],selector2)):
                # 			yield i
                # 			if D: self.debug("appended")
                # 		if D: self.debug("discarded")
                # 	except Exception as e:
                # 		if D: self.debug("discarded, Exception: %s",color.bold(e))
                else:
                  try:
                    # TODO optimize an event when @ is not used. exe(selector1) can be cached
                    if exe((selector0, exe(selector1), exe(selector2))):
                      yield i
                      if D: self.debug("appended %s", i)
                    elif D: self.debug("discarded")
                  except Exception:
                    if D: self.debug("discarded")

            # if D and nodeList: self.debug("returning '%s' objects: '%s'", color.bold(len(nodeList)), color.bold(nodeList))
            return exeSelector(fst)
          self.current = fst
          snd = exe(node[2])
          typefst = type(fst)
          if typefst in [tuple] + ITER_TYPES + STR_TYPES:
            typesnd = type(snd)
            # nodes[N]
            if typesnd in NUM_TYPES or typesnd is str and snd.isdigit():
              n = int(snd)
              if D:
                self.info(
                    "getting %sth element from '%s'", color.bold(n),
                    color.bold(fst)
                )
              if typefst in (generator, chain):
                if n > 0:
                  return skip(fst, n)
                elif n == 0:
                  return next(fst)
                else:
                  fst = list(fst)
              else:
                try:
                  return fst[n]
                except (IndexError, TypeError):
                  return None
            # $.*['string']==$.string
            if type(snd) in STR_TYPES:
              return exe((".", fst, snd))
            else:
              # $.*[@.string] - bad syntax, but allowed
              return snd
          else:
            try:
              if D: self.debug("returning %s", color.bold(fst[snd]))
              return fst[snd]
            except KeyError:
              # CHECK - is it ok to do that or should it be ProgrammingError?
              if D: self.debug("returning an empty list")
              return []
        raise ProgrammingError(
            "Wrong usage of " + color.bold("[") + " operator"
        )
      elif op == "fn":
        # Built-in functions
        fnName = node[1]
        args = None
        try:
          args = [exe(x) for x in node[2:]]
        except IndexError:
          if D:
            self.debug("NOT ERROR: can't map '%s' with '%s'", node[2:], exe)
        # arithmetic
        if fnName == "sum":
          args = args[0]
          if type(args) in NUM_TYPES:
            return args
          return sum((x for x in args if type(x) in NUM_TYPES))
        elif fnName == "max":
          args = args[0]
          if type(args) in NUM_TYPES:
            return args
          return max((x for x in args if type(x) in NUM_TYPES))
        elif fnName == "min":
          args = args[0]
          if type(args) in NUM_TYPES:
            return args
          return min((x for x in args if type(x) in NUM_TYPES))
        elif fnName == "avg":
          args = args[0]
          if type(args) in NUM_TYPES:
            return args
          if type(args) not in ITER_TYPES:
            raise Exception("Argument for avg() is not an array")
          else:
            args = list(args)
          try:
            return sum(args)/float(len(args))
          except TypeError:
            args = [x for x in args if type(x) in NUM_TYPES]
            self.warning("Some items in array were ommited")
            return sum(args)/float(len(args))
        elif fnName == "round":
          return round(*args)
        # casting
        elif fnName == "int":
          return int(args[0])
        elif fnName == "float":
          return float(args[0])
        elif fnName == "str":
          return str(py2JSON(args[0]))
        elif fnName in ("list", "array"):
          try:
            a = args[0]
          except IndexError:
            return []
          targs = type(a)
          if targs is timeutils.datetime.datetime:
            return timeutils.date2list(a) + timeutils.time2list(a)
          if targs is timeutils.datetime.date:
            return timeutils.date2list(a)
          if targs is timeutils.datetime.time:
            return timeutils.time2list(a)
          return list(a)
        # string
        elif fnName == "upper":
          return args[0].upper()
        elif fnName == "lower":
          return args[0].lower()
        elif fnName == "capitalize":
          return args[0].capitalize()
        elif fnName == "title":
          return args[0].title()
        elif fnName == "split":
          return args[0].split(*args[1:])
        elif fnName == "slice":
          if args and type(args[1]) not in ITER_TYPES:
            raise ExecutionError(
                "Wrong usage of slice(STRING, ARRAY). Second argument is not an array but %s."
                % color.bold(type(args[1]).__name__)
            )
          try:
            pos = list(args[1])
            if type(pos[0]) in ITER_TYPES:
              if D: self.debug("run slice() for a list of slicers")
              return (args[0][x[0]:x[1]] for x in pos)
            return args[0][pos[0]:pos[1]]
          except IndexError:
            if len(args) != 2:
              raise ProgrammingError(
                  "Wrong usage of slice(STRING, ARRAY). Provided %s argument, should be exactly 2."
                  % len(args)
              )
        elif fnName == "escape":
          global escape, escapeDict
          if not escape:
            from objectpath.utils import escape, escapeDict
          return escape(args[0], escapeDict)
        elif fnName == "unescape":
          global unescape, unescapeDict
          if not unescape:
            from objectpath.utils import unescape, unescapeDict
          return unescape(args[0], unescapeDict)
        elif fnName == "replace":
          if sys.version_info[0] < 3 and type(args[0]) is unicode:
            args[0] = args[0].encode("utf8")
          return str.replace(args[0], args[1], args[2])
        # TODO this should be supported by /regex/
        # elif fnName=="REsub":
        # 	return re.sub(args[1],args[2],args[0])
        elif fnName == "sort":
          if len(args) > 1:
            key = args[1]
            a = {"key": lambda x: x.get(key, 0)}
          else:
            a = {}
          args = args[0]
          if D: self.debug("doing sort on '%s'", args)
          try:
            return sorted(args, **a)
          except TypeError:
            return args
        elif fnName == "reverse":
          args = args[0]
          try:
            args.reverse()
            return args
          except TypeError:
            return args
        elif fnName == "unique":
          try:
            return list(set(args[0]))
          except TypeError:
            return args[0]
        elif fnName == "map":
          return chain.from_iterable(map(lambda x: exe(("fn", args[0], x)), args[1]))
        elif fnName in ("count", "len"):
          args = args[0]
          if args in (True, False, None):
            return args
          if type(args) in ITER_TYPES:
            return len(list(args))
          return len(args)
        elif fnName == "join":
          try:
            joiner = args[1]
          except Exception:
            joiner = ""
          try:
            return joiner.join(args[0])
          except TypeError:
            try:
              return joiner.join(map(str, args[0]))
            except Exception:
              return args[0]
        # time
        elif fnName in ("now", "age", "time", "date", "dateTime"):
          if fnName == "now":
            return timeutils.now()
          if fnName == "date":
            return timeutils.date(args)
          if fnName == "time":
            return timeutils.time(args)
          if fnName == "dateTime":
            return timeutils.dateTime(args)
          # TODO move lang to localize() entirely!
          if fnName == "age":
            a = {}
            if len(args) > 1:
              a["reference"] = args[1]
            if len(args) > 2:
              a["lang"] = args[2]
            return list(timeutils.age(args[0], **a))
        elif fnName == "toMillis":
          args = args[0]
          if args.utcoffset() is not None:
            args = args - args.utcoffset()  # pylint: disable=E1103
          global calendar
          if not calendar:
            import calendar
          return int(
              calendar.timegm(args.timetuple())*1000 + args.microsecond/1000
          )
        elif fnName == "localize":
          if type(args[0]) is timeutils.datetime.datetime:
            return timeutils.UTC2local(*args)
        # polygons
        elif fnName == "area":

          def segments(p):
            p = list(map(lambda x: x[0:2], p))
            return zip(p, p[1:] + [p[0]])

          return 0.5*abs(
              sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in segments(args[0]))
          )
        # misc
        elif fnName == "keys":
          try:
            return list(args[0].keys())
          except AttributeError:
            raise ExecutionError(
                "Argument is not " + color.bold("object") +
                " but %s in keys()" % color.bold(type(args[0]).__name__)
            )
        elif fnName == "values":
          try:
            return list(args[0].values())
          except AttributeError:
            raise ExecutionError(
                "Argument is not " + color.bold("object") +
                " but %s in values()" % color.bold(type(args[0]).__name__)
            )
        elif fnName == "type":
          ret = type(args[0])
          if ret in ITER_TYPES:
            return "array"
          if ret is dict:
            return "object"
          return ret.__name__
        elif fnName in self._REGISTERED_FUNCTIONS:
          return self._REGISTERED_FUNCTIONS[fnName](*args)
        else:
          raise ProgrammingError(
              "Function " + color.bold(fnName) + " does not exist."
          )
      else:
        return node

    D = self.D
    if type(expr) in STR_TYPES:
      tree = self.compile(expr)
    elif type(expr) not in (tuple, list, dict):
      return expr
    ret = exe(tree)
    if D: self.end("Tree.execute with: %s", color.bold(self.cleanOutput(ret)))
    return ret

    def delete(self, path_str):
      """ Returns True if data at any path is deleted or the path is missing """
      self._locked = False
      try:
        return self._update(path_str, None, delete=True)
      finally:
        self._locked = True

  def update(self, path_str, newvalue):
    """ Returns True if the tree is updated due to this operation.
      Multiple elements can be modified if selectors are used in path"""
    self._locked = False
    try:
      return self._update(path_str, newvalue, delete=False)
    finally:
      self._locked = True

  def delete(self, path_str):
    """ Returns True if data at any path is deleted or the path is missing """
    self._locked = False
    try:
      return self._update(path_str, None, delete=True)
    finally:
      self._locked = True

  def _update(self, path_str, newvalue, delete):

    def validate_new_value():
      """
      TODO:
      If the newly added value to be validated, or modified before setting.
      eq. Convert None to {} for root data or convert tuples and sets to list
      """
      return True, newvalue

    if delete is not True:
      r, newvalue = validate_new_value()
      if not r:
        raise ValueError("Invalid value: {}".format(str(newvalue)))

    if path_str and type(path_str) in STR_TYPES:
      path = path_str.strip()

      if path == "$":
        """ Special case. Update the whole tree """
        self.sanitize_tree_state()
        if delete is True:
          self.data = {}
          return True
        self.data = newvalue
        return True
      else:
        try:
          tree = self.compile(path)
        except Exception as e:
          raise InvalidPath("Objectpath parsing failed with exception {}: {}".format(type(e), str(e)))

        if len(tree) < 3:
          raise InvalidPath("Objectpath parsing failed. Parse tree is {}".format(str(tree)))

        try:
          # Find if we have a single parent or more
          parent_path = create_path_from_tree(tree[1])
          parent = self.execute(parent_path)

          if parent is None:
            # Means broken path
            # If the path is creatable under the current tree, do it
            parent = attempt_creation(parent_path)
            if not parent:
              if delete is True:
                return True
              raise InvalidPath("Objectpath value update failed for path: {}. {}".format(
                parent_path, "Unable to create path for current data"))

          if isinstance(parent, generator):
            # Means some selector is involved. Can have multiple hits
            parents = list(parent)
          else:
            # Hopefully, straightforward addressing. Single path to patch
            parents = [parent]

          if not parents:
            return delete is True

          child_path = create_path_from_tree(tree[2])

          if tree[0] == ".":
            """
            Format: "$.<optional-ancestors>.child"
            """
            exp = None
            modified = False

            for parent in parents:
              try:
                if delete is True:
                  if child_path in parent:
                    del parent[child_path]
                    modified = True
                else:
                  parent[child_path] = newvalue
                  modified = True
              except Exception as e:
                if not exp:
                  exp = e
            if modified:
              self.sanitize_tree_state()
            if exp:
              if delete is True:
                return True
              raise InvalidPath(
                "Objectpath value update failed with exception {}: {}".format(type(exp), str(exp)))
            return modified
          elif tree[0] == "[":
            """
            Format: "$.<optional-ancestors>.child[0] or
                "$.<optional-ancestors>.child[@condition]
            """

            # Now find out if the content inside the bracket can be used as a key or an index

            def is_selector(child_tree):
              if isinstance(child_tree, tuple) and child_tree:
                if (len(child_tree) > 1) and (child_tree[0] in SELECTOR_OPS):
                  return True
              return False

            if not is_selector(tree[2]):
              child_value = self.execute(child_path)

              exp = None
              modified = False

              for parent in parents:
                try:
                  if delete is True:
                    if isinstance(parent, list):
                      if isinstance(child_value, int) and len(parent) > child_value:
                        del parent[child_value]
                    else:
                      if child_value in parent:
                        del parent[child_value]
                    modified = True
                  else:
                    if isinstance(parent, list) and len(parent) == child_value:
                        parent.append(newvalue)
                        modified = True
                    else:
                      parent[child_value] = newvalue
                      modified = True
                except Exception as e:
                  exp = e

              if modified:
                self.sanitize_tree_state()
              if exp:
                if delete is True:
                  return True
                raise InvalidPath(
                  "Objectpath value update failed with exception {}: {}".format(type(exp), str(exp)))
              return (delete is True) or modified
            else:
              targets = []
              target = self.execute(path)
              if isinstance(target, generator):
                targets = list(target)
              elif target is not None:
                targets = [target]

              if not targets:
                return delete is True

              g_parents = []
              parent_tree = self.compile(parent_path)

              if parent_path == "$":
                g_parent = self.data
              else:
                """ TODO: Next line should break for some paths. Find and handle them """
                g_parent = self.execute(create_path_from_tree(parent_tree[1]))

              if isinstance(g_parent, generator):
                g_parents = list(g_parent)
              elif g_parent is not None:
                g_parents = [g_parent]
              else:
                g_parents = [self.data]

              exp = None
              modified = False

              for parent in parents:
                if isinstance(parent, list):
                  for target in targets:
                    if target != newvalue:
                      while True:
                        try:
                          i = parent.index(target)
                          if delete is True:
                            del parent[i]
                          else:
                            parent[i] = newvalue
                          modified = True
                        except ValueError as e:
                          break
                elif isinstance(parent, dict):
                  if parent != newvalue:
                    for g_parent in g_parents:
                      if isinstance(g_parent, list):
                        while True:
                          try:
                            i = g_parent.index(parent)
                            if delete is True:
                              del g_parent[i]
                            else:
                              g_parent[i] = newvalue
                            modified = True
                          except ValueError as e:
                            break
              if modified:
                self.sanitize_tree_state()
              return (delete is True) or modified
        except Exception as e:
          raise InvalidPath("Objectpath splitting failed with exception {}: {}".format(type(e), str(e)))

    else:
      raise InvalidPath("Invalid path: {}".format(str(path_str)))

    return delete is True

  def __str__(self):
    return "TreeObject()"

  def __repr__(self):
    return self.__str__()
