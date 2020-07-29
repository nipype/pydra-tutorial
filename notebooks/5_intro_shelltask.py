# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
import nest_asyncio
nest_asyncio.apply()
# -

# ## ShellCommandTask
#
# In addition to `FunctionTask`, pydra allows for creating tasks from shell commands by using `ShellCommandTask`.
#
# Let's run a simple command `pwd` using pydra

import pydra

# +
cmd = "pwd"
# we should use an executable to pass the command we want to run
shelly = pydra.ShellCommandTask(name="shelly", executable=cmd)

# we can always check the cmdline of our task
shelly.cmdline
# -

# and now let's try to run it:

with pydra.Submitter(plugin="cf") as sub:
    sub(shelly)


# and check the result

shelly.result()

# the result should have `return_code`, `stdout` and `stderr`. If everything goes well `return_code` should be `0`, `stdout` should point to the working directory and `stderr` should be an empty string.

# ### Commands with arguments and inputs
# you can also use longer commands by providing a list:

# +
cmd = ["echo", "hail", "pydra"]
shelly = pydra.ShellCommandTask(name="shelly", executable=cmd)
print("cmndline = ", shelly.cmdline)

with pydra.Submitter(plugin="cf") as sub:
    sub(shelly)
shelly.result()
# -

# #### using args
# In addition to `executable`, we can also use `args`. The previous example can be also rewritten as:

# +
cmd = "echo"
args = ["hail", "pydra"]

shelly = pydra.ShellCommandTask(name="shelly", executable=cmd, args=args)
print("cmndline = ", shelly.cmdline)

with pydra.Submitter(plugin="cf") as sub:
    sub(shelly)
shelly.result()
# -

# ### Customized input
#
# Pydra always checks `executable` and `args`, but we can also provide additional inputs, in order to do it, we have to modify `input_spec` first by using `SpecInfo` class:

# +
import attr

my_input_spec = pydra.specs.SpecInfo(
    name="Input",
    fields=[
        (
            "text",
            attr.ib(
                type=str,
                metadata={"position": 1, "argstr": "", "help_string": "text", "mandatory": True},
                ),
        )
    ],
    bases=(pydra.specs.ShellSpec,),
)

# -

# Notice, that in order to create your own `input_spec`, you have to provide a list of `fields`. There are several valid syntax options to specify elements of `fields`:
# - `(name, attribute)`
# - `(name, type, default)`
# - `(name, type, default, metadata)`
# - `(name, type, metadata)`
#
# where `name`, `type`, and `default` are the name, type and default values of the field. `attribute` is defined by using `attr.ib`, in the example the attribute has `type` and `metadata`, but the full specification can be found [here](https://www.attrs.org/en/stable/api.html#attr.ib).
#
# In `metadata`, you can provide additional information that is used by `pydra`, `help_string` is the only key that is required, and the full list of supported keys is `['position', 'argstr', 'requires', 'mandatory', 'allowed_values', 'output_field_name', 'copyfile', 'separate_ext', 'container_path', 'help_string', 'xor', 'output_file_template']`. Among the supported keys, you have:
# - `help_string`: a sring, description of the argument;
# - `position`: integer greater than 0, defines the relative position of the arguments when the shell command is constructed;
# - `argstr`: a string, e.g. "-o", can be used to specify a flag if needed for the command argument;
# - `mandatory`: a bool, if `True` pydra will raise an exception, if the argument is not provided;
#
#

# To define `my_input_spec` we used the most general syntax that requires `(name, attribute)`, but
# perhaps the simplest syntax is the last one, that contains `(name, type, metadata)`. Using this syntax, `my_input_spec` could look like this:
#
# ```
# my_input_spec_short = pydra.specs.SpecInfo(
#     name="Input",
#     fields=[
#         ("text", str, {"position": 1, "help_string": "text", "mandatory": True}),
#     ],
#     bases=(pydra.specs.ShellSpec,),
# )
# ```

# After defining `my_input_spec`, we can define our task:

# +
cmd_exec = "echo"
hello = "HELLO"
shelly = pydra.ShellCommandTask(
    name="shelly", executable=cmd_exec, text=hello, input_spec=my_input_spec
)

print("cmndline = ", shelly.cmdline)

with pydra.Submitter(plugin="cf") as sub:
    sub(shelly)
shelly.result()
# -

# ### Customized output
#
# We can also customize the output if we want to return something more than the `stdout`, e.g. a file.

my_output_spec = pydra.specs.SpecInfo(
    name="Output",
    fields=[("newfile", pydra.specs.File, "newfile_tmp.txt")],
    bases=(pydra.specs.ShellOutSpec,),
)


# now we can create a task that returns a new file:

# +
cmd = ["touch", "newfile_tmp.txt"]
shelly = pydra.ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

print("cmndline = ", shelly.cmdline)

with pydra.Submitter(plugin="cf") as sub:
    sub(shelly)
shelly.result()

# + [markdown] solution2="hidden" solution2_first=true
# #### exercise 1
#
# Write a task that creates two new files, use the provided output spec.

# + solution2="hidden"
cmd = "echo"
args = ["newfile_1.txt", "newfile_2.txt"]

my_output_spec = pydra.specs.SpecInfo(
    name="Output",
    fields=[
        (
            "out1",
            attr.ib(
                type=pydra.specs.File,
                metadata={
                    "output_file_template": "{args}",
                    "help_string": "output file",
                },
            ),
        )
    ],
    bases=(pydra.specs.ShellOutSpec,),
)


shelly = pydra.ShellCommandTask(name="shelly", executable=cmd, args=args, output_spec=my_output_spec).split("args")

print("cmndline = ", shelly.cmdline)

with pydra.Submitter(plugin="cf") as sub:
    sub(shelly)
shelly.result()

# +
cmd = "echo"
args = ["newfile_1.txt", "newfile_2.txt"]

my_output_spec = pydra.specs.SpecInfo(
    name="Output",
    fields=[
        (
            "out1",
            attr.ib(
                type=pydra.specs.File,
                metadata={
                    "output_file_template": "{args}",
                    "help_string": "output file",
                },
            ),
        )
    ],
    bases=(pydra.specs.ShellOutSpec,),
)

# write your solution here
# -

# # DO NOT RUN IF Docker IS NOT AVAILABLE
#
# **Note, that the following tasks use Docker, so they will fail if Docker is not available. It will also fail in Binder.**

# ### DockerTask
#
# all the commands can also be run in a docker container using `DockerTask`. The syntax is very similar, but an additional argument `image` is required.

# + tags=["raises-exception"]
cmd = "whoami"
docky = pydra.DockerTask(name="docky", executable=cmd, image="busybox")

with pydra.Submitter() as sub:
    docky(submitter=sub)

docky.result()

# + [markdown] solution2="shown" solution2_first=true
# #### exercise2
#
# Use splitter to run the same command in two different images:

# + solution2="shown" tags=["raises-exception"]
cmd = "whoami"
docky = pydra.DockerTask(name="docky", executable=cmd, image=["busybox", "ubuntu"]).split("image")

with pydra.Submitter() as sub:
    docky(submitter=sub)

docky.result()

# +
# write your solution here
# -

# #### Using `ShellCommandTask` with `container_info` argument:
#
# You can run the shell command in a docker container by adding the `container_info` argument to `ShellCommandTask`:

# + tags=["raises-exception"]
shelly = pydra.ShellCommandTask(name="shelly", executable="whoami", container_info=("docker", "busybox"))
with pydra.Submitter() as sub:
    shelly(submitter=sub)

shelly.result()
# -

# If we don't provide `container_info` the output should be different:

# +
shelly = pydra.ShellCommandTask(name="shelly", executable="whoami")
with pydra.Submitter() as sub:
    shelly(submitter=sub)

shelly.result()
# -
