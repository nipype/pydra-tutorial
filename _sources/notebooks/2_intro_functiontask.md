---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# FunctionTask

```{code-cell}
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

```{code-cell}
import pydra


@pydra.mark.task
def add_var(a, b):
    return a + b
```

Once we decorate the function, we can create a pydra `Task` and specify the input:

```{code-cell}
task1 = add_var(a=4, b=5)
```

We can check the type of `task1`:

```{code-cell}
type(task1)
```

and we can check if the task has correct values of `a` and `b`, they should be saved in the task `inputs`:

```{code-cell}
print(f'a = {task1.inputs.a}')
print(f'b = {task1.inputs.b}')
```

We can also check content of entire `inputs`:

```{code-cell}
task1.inputs
```

As you could see, `task.inputs` contains also information about the function, that is an inseparable part of the `FunctionTask`.

Once we have the task with set input, we can run it. Since `Task` is a "callable object", we can use the syntax:

```{code-cell}
task1()
```

As you can see, the result was returned right away, but we can also access it later:

```{code-cell}
task1.result()
```

`Result` contains more than just an output, so if we want to get the task output, we can type:

```{code-cell}
result = task1.result()
result.output.out
```

And if we want to see the input that was used in the task, we can set an optional argument `return_inputs` to True.

```{code-cell}
task1.result(return_inputs=True)
```

## Customizing output names
Note, that "out" is the default name for the task output, but we can always customize it. There are two ways of doing it: using *python* function annotation and using another *pydra* decorator:

Let's start from the function annotation:

```{code-cell}
import typing as ty


@pydra.mark.task
def add_var_an(a, b) -> {'sum_a_b': int}:
    return a + b


task1a = add_var_an(a=4, b=5)
task1a()
```

The annotation might be very useful to specify the output names when the function returns multiple values.

```{code-cell}
@pydra.mark.task
def modf_an(a) -> {'fractional': ty.Any, 'integer': ty.Any}:
    import math

    return math.modf(a)


task2 = modf_an(a=3.5)
task2()
```

The second way of customizing the output requires another decorator - `pydra.mark.annotate`

```{code-cell}
@pydra.mark.task
@pydra.mark.annotate({'return': {'fractional': ty.Any, 'integer': ty.Any}})
def modf(a):
    import math

    return math.modf(a)


task2a = modf(a=3.5)
task2a()
```

**Note, that the order of the pydra decorators is important!**

+++

## Setting the input

We don't have to provide the input when we create a task, we can always set it later:

```{code-cell}
task3 = add_var()
task3.inputs.a = 4
task3.inputs.b = 5
task3()
```

If we don't specify the input, `attr.NOTHING` will be used as the default value

```{code-cell}
task3a = add_var()
task3a.inputs.a = 4

# importing attr library, and checking the type of `b`
import attr

task3a.inputs.b == attr.NOTHING
```

And if we try to run the task, an error will be raised:

```{code-cell}
:tags: [raises-exception]

task3a()
```

## Output directory and caching the results

After running the task, we can check where the output directory with the results was created:

```{code-cell}
task3.output_dir
```

Within the directory you can find the file with the results: `_result.pklz`.

```{code-cell}
import os
```

```{code-cell}
os.listdir(task3.output_dir)
```

But we can also provide the path where we want to store the results. If a path is provided for the cache directory, then pydra will use the cached results of a node instead of recomputing the result. Let's create a temporary directory and a specific subdirectory "task4":

```{code-cell}
from tempfile import mkdtemp
from pathlib import Path
```

```{code-cell}
cache_dir_tmp = Path(mkdtemp()) / 'task4'
print(cache_dir_tmp)
```

Now we can pass this path to the argument of `FunctionTask` - `cache_dir`. To observe the execution time, we specify a function that is sleeping for 5s:

```{code-cell}
@pydra.mark.task
def add_var_wait(a, b):
    import time

    time.sleep(5)
    return a + b


task4 = add_var_wait(a=4, b=6, cache_dir=cache_dir_tmp)
```

If you're running the cell first time, it should take around 5s.

```{code-cell}
task4()
task4.result()
```

We can check `output_dir` of our task, it should contain the path of `cache_dir_tmp` and the last part contains the name of the task class `FunctionTask` and the task checksum:

```{code-cell}
task4.output_dir
```

Let's see what happens when we defined identical task again with the same `cache_dir`:

```{code-cell}
task4a = add_var_wait(a=4, b=6, cache_dir=cache_dir_tmp)
task4a()
```

This time the result should be ready right away! *pydra* uses available results and do not recompute the task.

*pydra* not only checks for the results in `cache_dir`, but you can provide a list of other locations that should be checked. Let's create another directory that will be used as `cache_dir` and previous working directory will be used in `cache_locations`.

```{code-cell}
cache_dir_tmp_new = Path(mkdtemp()) / 'task4b'

task4b = add_var_wait(
    a=4, b=6, cache_dir=cache_dir_tmp_new, cache_locations=[cache_dir_tmp]
)
task4b()
```

This time the results should be also returned quickly! And we can check that `task4b.output_dir` was not created:

```{code-cell}
task4b.output_dir.exists()
```

If you want to rerun the task regardless having already the results, you can set `rerun` to `True`. The task will take several seconds and new `output_dir` will be created:

```{code-cell}
cache_dir_tmp_new = Path(mkdtemp()) / 'task4c'

task4c = add_var_wait(
    a=4, b=6, cache_dir=cache_dir_tmp_new, cache_locations=[cache_dir_tmp]
)
task4c(rerun=True)

task4c.output_dir.exists()
```

If we update the input of the task, and run again, the new directory will be created and task will be recomputed:

```{code-cell}
task4b.inputs.a = 1
print(task4b())
print(task4b.output_dir.exists())
```

and when we check the `output_dir`, we can see that it's different than last time:

```{code-cell}
task4b.output_dir
```

This is because, the checksum changes when we change either input or function.

+++ {"solution2": "hidden", "solution2_first": true}

### Exercise 1
Create a task that take a list of numbers as an input and returns two fields: `mean` with the mean value and `std` with the standard deviation value.

```{code-cell}
:tags: [hide-cell]

@pydra.mark.task
@pydra.mark.annotate({'return': {'mean': ty.Any, 'std': ty.Any}})
def mean_dev(my_list):
    import statistics as st

    return st.mean(my_list), st.stdev(my_list)


my_task = mean_dev(my_list=[2, 2, 2])
my_task()
my_task.result()
```

```{code-cell}
# write your solution here (you can use statistics module)
```

## Using Audit

*pydra* can record various run time information, including the workflow provenance, by setting `audit_flags` and the type of messengers.

`AuditFlag.RESOURCE` allows you to monitor resource usage for the `Task`, while `AuditFlag.PROV` tracks the provenance of the `Task`.

```{code-cell}
from pydra.utils.messenger import AuditFlag, PrintMessenger

task5 = add_var(a=4, b=5, audit_flags=AuditFlag.RESOURCE)
task5()
task5.result()
```

One can turn on both audit flags using `AuditFlag.ALL`, and print the messages on the terminal using the `PrintMessenger`.

```{code-cell}
task5 = add_var(
    a=4, b=5, audit_flags=AuditFlag.ALL, messengers=PrintMessenger()
)
task5()
task5.result()
```
