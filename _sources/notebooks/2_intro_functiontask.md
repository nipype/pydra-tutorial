---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# FunctionTask

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
import nest_asyncio

nest_asyncio.apply()
```

A `FunctionTask` is a `Task` that can be created from every *python* function by using *pydra* decorator: `pydra.mark.task`:

```{code-cell} ipython3
import pydra

@pydra.mark.task
def add_var(a, b):
    return a + b
```

Once we decorate the function, we can create a pydra `Task` and specify the input:

```{code-cell} ipython3
task0 = add_var(a=4, b=5)
```

We can check the type of `task0`:

```{code-cell} ipython3
type(task0)
```

and we can check if the task has correct values of `a` and `b`, they should be saved in the task `inputs`:

```{code-cell} ipython3
print(f'a = {task0.inputs.a}')
print(f'b = {task0.inputs.b}')
```

We can also check content of entire `inputs`:

```{code-cell} ipython3
task0.inputs
```

As you could see, `task.inputs` contains also information about the function, that is an inseparable part of the `FunctionTask`.

Once we have the task with set input, we can run it. Since `Task` is a "callable object", we can use the syntax:

```{code-cell} ipython3
task0()
```

As you can see, the result was returned right away, but we can also access it later:

```{code-cell} ipython3
task0.result()
```

`Result` contains more than just an output, so if we want to get the task output, we can type:

```{code-cell} ipython3
result = task0.result()
result.output.out
```

And if we want to see the input that was used in the task, we can set an optional argument `return_inputs` to True.

```{code-cell} ipython3
task0.result(return_inputs=True)
```

## Type-checking

+++

### What is Type-checking?

Type-checking is verifying the type of a value at compile or run time. It ensures that operations or assignments to variables are semantically meaningful and can be executed without type errors, enhancing code reliability and maintainability.

+++

### Why Use Type-checking?

1. **Error Prevention**: Type-checking helps catch type mismatches early, preventing potential runtime errors.
2. **Improved Readability**: Type annotations make understanding what types of values a function expects and returns more straightforward.
3. **Better Documentation**: Explicitly stating expected types acts as inline documentation, simplifying code collaboration and review.
4. **Optimized Performance**: Type-related optimizations can be made during compilation when types are explicitly specified.

+++

### How is Type-checking Implemented in Pydra?

+++

#### Static Type-Checking
Static type-checking is done using Python's type annotations. You annotate the types of your function arguments and the return type and then use a tool like `mypy` to statically check if you're using the function correctly according to those annotations.

```{code-cell} ipython3
@pydra.mark.task
def add(a: int, b: int) -> int:
    return a + b
```

```{code-cell} ipython3
# This usage is correct according to static type hints:
task1a = add(a=5, b=3)
task1a()
```

```{code-cell} ipython3
:tags: [raises-exception]
# This usage is incorrect according to static type hints:
task1b = add(a="hello", b="world")
task1b()
```

#### Dynamic Type-Checking

Dynamic type-checking is done at runtime. Add dynamic type checks if you want to enforce types when the function is executed.

```{code-cell} ipython3
@pydra.mark.task
def add(a, b):
    if not (isinstance(a, int) and isinstance(b, int)):
        raise TypeError("Both inputs should be integers.")
    return a + b
```

```{code-cell} ipython3
# This usage is correct and will not raise a runtime error:
task1c = add(a=5, b=3)
task1c()
```

```{code-cell} ipython3
:tags: [raises-exception]
# This usage is incorrect and will raise a runtime TypeError:
task1d = add(a="hello", b="world")
task1d()
```

#### Checking Complex Types

For more complex types like lists, dictionaries, or custom objects, we can use type hints combined with dynamic checks.

```{code-cell} ipython3
from typing import List, Tuple

@pydra.mark.task
def sum_of_pairs(pairs: List[Tuple[int, int]]) -> List[int]:
    if not all(isinstance(pair, Tuple) and len(pair) == 2 for pair in pairs):
        raise ValueError("Input should be a list of pairs (tuples with 2 integers each).")
    return [sum(pair) for pair in pairs]
```

```{code-cell} ipython3
# Correct usage
task1e = sum_of_pairs(pairs=[(1, 2), (3, 4)])  
task1e()
```

```{code-cell} ipython3
:tags: [raises-exception]
# This will raise a ValueError
task1f = sum_of_pairs(pairs=[(1, 2), (3, "4")])  
task1f()
```

## Customizing output names
Note, that "out" is the default name for the task output, but we can always customize it. There are two ways of doing it: using *python* function annotation and using another *pydra* decorator:

Let's start from the function annotation:

```{code-cell} ipython3
import typing as ty

@pydra.mark.task
def add_var_an(a: int, b: int) -> {'sum_a_b': int}:
    return a + b


task2a = add_var_an(a=4, b=5)
task2a()
```

The annotation might be very useful to specify the output names when the function returns multiple values.

```{code-cell} ipython3
@pydra.mark.task
def modf_an(a: float) -> {'fractional': ty.Any, 'integer': ty.Any}:
    import math

    return math.modf(a)


task2b = modf_an(a=3.5)
task2b()
```

The second way of customizing the output requires another decorator - `pydra.mark.annotate`

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate({'return': {'fractional': ty.Any, 'integer': ty.Any}})
def modf(a: float):
    import math

    return math.modf(a)

task2c = modf(a=3.5)
task2c()
```

**Note, that the order of the pydra decorators is important!**

+++

## Setting the input

We don't have to provide the input when we create a task, we can always set it later:

```{code-cell} ipython3
task3 = add_var()
task3.inputs.a = 4
task3.inputs.b = 5
task3()
```

If we don't specify the input, `attr.NOTHING` will be used as the default value

```{code-cell} ipython3
task3a = add_var()
task3a.inputs.a = 4

# importing attr library, and checking the type of `b`
import attr

task3a.inputs.b == attr.NOTHING
```

And if we try to run the task, an error will be raised:

```{code-cell} ipython3
:tags: [raises-exception]

task3a()
```

## Output directory and caching the results

After running the task, we can check where the output directory with the results was created:

```{code-cell} ipython3
task3.output_dir
```

Within the directory you can find the file with the results: `_result.pklz`.

```{code-cell} ipython3
import os
```

```{code-cell} ipython3
os.listdir(task3.output_dir)
```

But we can also provide the path where we want to store the results. If a path is provided for the cache directory, then pydra will use the cached results of a node instead of recomputing the result. Let's create a temporary directory and a specific subdirectory "task4":

```{code-cell} ipython3
from tempfile import mkdtemp
from pathlib import Path
```

```{code-cell} ipython3
cache_dir_tmp = Path(mkdtemp()) / 'task4'
print(cache_dir_tmp)
```

Now we can pass this path to the argument of `FunctionTask` - `cache_dir`. To observe the execution time, we specify a function that is sleeping for 5s:

```{code-cell} ipython3
@pydra.mark.task
def add_var_wait(a: int, b: int):
    import time

    time.sleep(5)
    return a + b

task4 = add_var_wait(a=4, b=6, cache_dir=cache_dir_tmp)
```

If you're running the cell first time, it should take around 5s.

```{code-cell} ipython3
task4()
task4.result()
```

We can check `output_dir` of our task, it should contain the path of `cache_dir_tmp` and the last part contains the name of the task class `FunctionTask` and the task checksum:

```{code-cell} ipython3
task4.output_dir
```

Let's see what happens when we defined identical task again with the same `cache_dir`:

```{code-cell} ipython3
task4a = add_var_wait(a=4, b=6, cache_dir=cache_dir_tmp)
task4a()
```

This time the result should be ready right away! *pydra* uses available results and do not recompute the task.

*pydra* not only checks for the results in `cache_dir`, but you can provide a list of other locations that should be checked. Let's create another directory that will be used as `cache_dir` and previous working directory will be used in `cache_locations`.

```{code-cell} ipython3
cache_dir_tmp_new = Path(mkdtemp()) / 'task4b'

task4b = add_var_wait(
    a=4, b=6, cache_dir=cache_dir_tmp_new, cache_locations=[cache_dir_tmp]
)
task4b()
```

This time the results should be also returned quickly! And we can check that `task4b.output_dir` was not created:

```{code-cell} ipython3
task4b.output_dir.exists()
```

If you want to rerun the task regardless having already the results, you can set `rerun` to `True`. The task will take several seconds and new `output_dir` will be created:

```{code-cell} ipython3
cache_dir_tmp_new = Path(mkdtemp()) / 'task4c'

task4c = add_var_wait(
    a=4, b=6, cache_dir=cache_dir_tmp_new, cache_locations=[cache_dir_tmp]
)
task4c(rerun=True)

task4c.output_dir.exists()
```

If we update the input of the task, and run again, the new directory will be created and task will be recomputed:

```{code-cell} ipython3
task4b.inputs.a = 1
print(task4b())
print(task4b.output_dir.exists())
```

and when we check the `output_dir`, we can see that it's different than last time:

```{code-cell} ipython3
task4b.output_dir
```

This is because, the checksum changes when we change either input or function.

+++ {"solution2": "hidden", "solution2_first": true}

### Exercise 1
Create a task that take a list of numbers as an input and returns two fields: `mean` with the mean value and `std` with the standard deviation value.

```{code-cell} ipython3
:tags: [hide-cell]

@pydra.mark.task
@pydra.mark.annotate({'return': {'mean': ty.Any, 'std': ty.Any}})
def mean_dev(my_list: List):
    import statistics as st

    return st.mean(my_list), st.stdev(my_list)

my_task = mean_dev(my_list=[2, 2, 2])
my_task()
my_task.result()
```

```{code-cell} ipython3
# write your solution here (you can use statistics module)
```

## Using Audit

*pydra* can record various run time information, including the workflow provenance, by setting `audit_flags` and the type of messengers.

`AuditFlag.RESOURCE` allows you to monitor resource usage for the `Task`, while `AuditFlag.PROV` tracks the provenance of the `Task`.

```{code-cell} ipython3
from pydra.utils.messenger import AuditFlag, PrintMessenger

task5 = add_var(a=4, b=5, audit_flags=AuditFlag.RESOURCE)
task5()
task5.result()
```

One can turn on both audit flags using `AuditFlag.ALL`, and print the messages on the terminal using the `PrintMessenger`.

```{code-cell} ipython3
task5 = add_var(
    a=4, b=5, audit_flags=AuditFlag.ALL, messengers=PrintMessenger()
)
task5()
task5.result()
```
