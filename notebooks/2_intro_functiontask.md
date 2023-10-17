---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

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

# FunctionTask

+++

In this tutorial, you will generate your initial *Pydra* `Task`, which is a fundamental *Pydra*'s component capable of processing data. You will start from a `FunctionTask`, a type of `Task` that can be created from every *python* function by using *Pydra* decorator: `pydra.mark.task`:

```{code-cell} ipython3
import pydra

@pydra.mark.task
def add_var(a, b):
    return a + b
```

After decorating the function, you can create a Pydra `Task` and specify the input. In this example, values for `a` and `b` are needed.

```{code-cell} ipython3
task0 = add_var(a=4, b=5)
```

You can now check if the task has correct values of `a` and `b`, they should be saved in the task `inputs`:

```{code-cell} ipython3
print(f'a = {task0.inputs.a}')
print(f'b = {task0.inputs.b}')
```

You can also check content of entire `inputs`:

```{code-cell} ipython3
task0.inputs
```

As you could see, `task.inputs` contains also information about the function, that is an inseparable part of the `FunctionTask`.

Once you have the task with set values of input, you can run it. Since `Task` is a "callable object", we can use the following syntax:

```{code-cell} ipython3
task0()
```

As you can see, the result was returned right away, but you can also access it later:

```{code-cell} ipython3
task0.result()
```

The function should return the `Result` object. `Result` contains more than just an output, so if you want to get the task output, we can type:

```{code-cell} ipython3
result = task0.result()
result.output.out
```

You can also see the input that was used to run the task by setting an optional argument `return_inputs` to True.

```{code-cell} ipython3
task0.result(return_inputs=True)
```

Notice that the full name of the input variables contains the name of the task!

+++

If you want to practice, change the values of `a` and `b` and run the task again. 

+++

## Customizing output names
Note, that `out` from `result.output.out` is the default name for the task output, but you can always customize it by using *python* function annotation.

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

## Setting the input

Note that you don't have to provide the input when you create a task, you can always set it later:

```{code-cell} ipython3
task3 = add_var()
task3.inputs.a = 4
task3.inputs.b = 5
task3()
```

If you don't specify the input, `attr.NOTHING` will be used as the default value

```{code-cell} ipython3
task3a = add_var()
task3a.inputs.a = 4

task3a.inputs.b
```

And if you try to run the task, an error will be raised:

```{code-cell} ipython3
:tags: [raises-exception]

task3a()
```

You can now try to fix the task and run it again.

+++

## Output directory and caching the results

After running the task, you can check where the output directory with the results was created:

```{code-cell} ipython3
task3.output_dir
```

Within the directory you can find the file with the results: `_result.pklz`.

```{code-cell} ipython3
import os
os.listdir(task3.output_dir)
```

But you can also provide the path where you want to store the results. 
**Note that if the same path is provided when you run the task again, Pydra will use the cached results instead of recomputing the result.** 

Let's create a temporary directory and a specific subdirectory "task4":

```{code-cell} ipython3
from tempfile import mkdtemp
from pathlib import Path
```

```{code-cell} ipython3
cache_dir_tmp = Path(mkdtemp()) / 'task4'
print(cache_dir_tmp)
```

Now you can pass this path to the argument of `FunctionTask` - `cache_dir`. To observe the execution time, you can specify a function that is sleeping for 5s:

```{code-cell} ipython3
@pydra.mark.task
def add_var_wait(a: int, b: int):
    import time

    time.sleep(5)
    return a + b

task4 = add_var_wait(a=4, b=6, cache_dir=cache_dir_tmp)
```

If you're running the cell first time, it should take around 5s.

You can meassure the exact time by using a special method from Jupyter by adding `%%time`.

```{code-cell} ipython3
%%time
task4()
task4.result()
```

You can check `output_dir` of our task, it should contain the path of `cache_dir_tmp` and the last part contains the name of the task class `FunctionTask` and the task checksum that is unique for a specific function and specific set of input values. You can read more about checksum here TODO-LINK

```{code-cell} ipython3
task4.output_dir
```

Let's see what happens when an identical task is run again with the same `cache_dir`:

```{code-cell} ipython3
%%time
task4a = add_var_wait(a=4, b=6, cache_dir=cache_dir_tmp)
task4a()
```

This time the result should be ready right away! *Pydra* uses available results and do not recompute the task. The wall time provided by `%%tinme` should be in milliseconds.

*Pydra* not only checks for the results in `cache_dir`, but you can provide a list of other locations that should be checked. Let's create another directory that will be used as `cache_dir` and previous working directory will be used in `cache_locations`.

```{code-cell} ipython3
cache_dir_tmp_new = Path(mkdtemp()) / 'task4b'

task4b = add_var_wait(
    a=4, b=6, cache_dir=cache_dir_tmp_new, cache_locations=[cache_dir_tmp]
)
task4b()
```

This time the results should be also returned quickly! And you can check that `task4b.output_dir` was not created:

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

Remember that if you update the input of the task, the new directory will be created and task will be recomputed!

```{code-cell} ipython3
task4b.inputs.a = 1
print(task4b())
print(task4b.output_dir.exists())
```

and when you check the `output_dir`, you can see that it's different than last time:

```{code-cell} ipython3
task4b.output_dir
```

This is because, the checksum changes when you change either input or function.

+++ {"solution2": "hidden", "solution2_first": true}

### Exercise 1
Now you can practice creating new tasks!

Create a task that take a list of numbers as an input and returns two fields: `mean` with the mean value and `std` with the standard deviation value.

```{code-cell} ipython3
:tags: [hide-cell]

#TODO-HIDE
@pydra.mark.task
@pydra.mark.annotate({'return': {'mean': ty.Any, 'std': ty.Any}})
def mean_dev(my_list):
    import statistics as st

    return st.mean(my_list), st.stdev(my_list)

my_task = mean_dev(my_list=[2, 2, 2])
my_task()
my_task.result()
```

```{code-cell} ipython3
# write your solution here (you can use statistics module)
```
