Search.setIndex({"docnames": ["about/cite_pydra", "about/team", "notebooks/1_intro_pydra", "notebooks/2_intro_functiontask", "notebooks/3_intro_functiontask_state", "notebooks/4_intro_workflow", "notebooks/5_intro_shelltask", "notebooks/6_firstlevel_glm_nilearn", "notebooks/7_twolevel_glm_nilearn", "notebooks/8_intro_workers", "welcome"], "filenames": ["about/cite_pydra.md", "about/team.md", "notebooks/1_intro_pydra.md", "notebooks/2_intro_functiontask.md", "notebooks/3_intro_functiontask_state.md", "notebooks/4_intro_workflow.md", "notebooks/5_intro_shelltask.md", "notebooks/6_firstlevel_glm_nilearn.md", "notebooks/7_twolevel_glm_nilearn.md", "notebooks/8_intro_workers.md", "welcome.md"], "titles": ["Cite Pydra", "Team", "<span class=\"section-number\">1. </span>Intro to Pydra", "<span class=\"section-number\">2. </span>FunctionTask", "<span class=\"section-number\">3. </span>Tasks with States", "<span class=\"section-number\">4. </span>Workflow", "<span class=\"section-number\">5. </span>ShellCommandTask", "<span class=\"section-number\">7. </span>First Level GLM (from Nilearn)", "<span class=\"section-number\">8. </span>Two-Level GLM (from Nilearn)", "<span class=\"section-number\">6. </span>Workers in Pydra", "Welcome"], "terms": {"todo": [0, 1], "lightweight": 2, "python": [2, 3, 7, 8, 9], "3": [2, 3, 4, 5, 7, 8], "7": [2, 3, 5], "dataflow": 2, "engin": [2, 3, 4, 5, 7, 8], "graph": [2, 4, 5], "construct": [2, 6, 8], "manipul": 2, "distribut": [2, 9], "execut": [2, 3, 6, 9], "design": 2, "gener": [2, 3, 4, 6, 7], "purpos": [2, 7], "support": [2, 6, 9], "analyt": 2, "ani": [2, 3, 4, 5, 7, 8], "scientif": 2, "domain": 2, "creat": [2, 3, 4, 5, 6], "nipyp": [2, 3], "help": 2, "build": 2, "reproduc": 2, "scalabl": 2, "reusabl": 2, "fulli": 2, "autom": 2, "proven": [2, 3], "track": [2, 3], "workflow": [2, 3, 9, 10], "The": [2, 3, 4, 5, 6], "power": [2, 4, 5], "li": 2, "eas": 2, "creation": 2, "complex": [2, 4], "multiparamet": 2, "map": [2, 4, 8], "reduc": [2, 8], "oper": 2, "us": [2, 4, 5, 7, 8, 9, 10], "global": [2, 7, 8], "cach": 2, "s": [2, 3, 4, 5, 6, 7, 9], "kei": [2, 4, 6, 8], "featur": 2, "ar": [2, 3, 4, 5, 6, 7, 8, 9], "consist": 2, "api": [2, 8], "split": [2, 5, 6, 7, 8], "combin": [2, 8], "semant": 2, "level": [2, 5, 10], "recomput": [2, 3], "container": 2, "environ": 2, "There": [2, 3, 6], "two": [2, 3, 4, 5, 6, 7, 10], "main": [2, 5, 10], "type": [2, 3, 4, 6, 7, 8], "also": [2, 3, 4, 5, 6, 7, 8], "can": [2, 3, 4, 5, 6, 7, 8, 10], "nest": 2, "These": 2, "current": [2, 7, 8, 9], "implement": 2, "connect": [2, 7, 8], "multipl": [2, 3, 5, 7], "withing": 2, "functiontask": [2, 4, 5, 6, 10], "wrapper": 2, "function": [2, 3, 4, 5, 7, 8], "shellcommandtask": [2, 10], "shell": [2, 6], "command": [2, 3], "containertask": 2, "run": [2, 3, 4, 5, 6, 10], "within": [2, 3, 4, 5, 10], "contain": [2, 3, 4, 5, 6, 8], "dockertask": 2, "docker": [2, 6], "singularitytask": 2, "singular": 2, "concurrentfutur": [2, 4], "slurm": [2, 9], "dask": [2, 9], "psi": [2, 9], "j": [2, 9], "befor": [2, 4, 8], "go": [2, 7, 10], "next": [2, 7], "notebook": [2, 5, 10], "let": [2, 3, 4, 5, 6, 7], "check": [2, 3, 4, 6, 10], "properli": 2, "instal": [2, 8, 10], "import": [2, 3, 4, 5, 6, 7, 8], "nest_asyncio": [3, 4, 5, 6, 7, 8], "appli": [3, 4, 5, 6, 7, 8], "A": [3, 8, 9], "task": [3, 6, 8, 9, 10], "from": [3, 4, 5, 6, 10], "everi": [3, 8], "pydra": [3, 4, 5, 6, 7, 8], "decor": 3, "mark": [3, 4, 5, 7, 8], "def": [3, 4, 5, 7, 8], "add_var": [3, 4], "b": [3, 4, 5], "return": [3, 4, 5, 6, 7, 8], "onc": [3, 4, 5], "we": [3, 4, 5, 6, 7, 8], "specifi": [3, 4, 5, 6, 7, 8], "task1": [3, 4], "4": [3, 4, 7, 8], "5": [3, 4, 5, 7, 8], "ha": [3, 4, 5, 6, 8], "correct": [3, 8], "valu": [3, 4, 5, 6, 8], "thei": [3, 5, 6, 7], "should": [3, 4, 5, 6, 7], "save": [3, 7], "print": [3, 4, 6, 7, 8], "f": [3, 4, 8], "content": 3, "entir": [3, 5], "_func": 3, "x80": 3, "x05": 3, "x95": 3, "xba": 3, "x01": 3, "x00": 3, "x8c": 3, "x17cloudpickl": 3, "cloudpickl": 3, "x94": 3, "x0e_make_funct": 3, "x93": 3, "h": 3, "r_builtin_typ": 3, "x08codetyp": 3, "x85": 3, "x94r": 3, "k": 3, "x02k": 3, "x00k": 3, "x03c": 3, "x0c": 3, "x97": 3, "x01z": 3, "x94n": 3, "x01a": 3, "x01b": 3, "x86": 3, "tmp": [3, 6, 7], "ipykernel_6599": 3, "214681761": 3, "py": [3, 4], "x07add_var": 3, "x94h": 3, "x0ek": 3, "x04c": 3, "x0b": 3, "xe0": 3, "x88q": 3, "x895": 3, "x80l": 3, "x94c": 3, "t": [3, 4, 6, 7, 8], "x0b__package__": 3, "x08__name__": 3, "x08__main__": 3, "x94unnnt": 3, "x12_function_setst": 3, "x18": 3, "x15h": 3, "x0e": 3, "x0c__qualname__": 3, "x0f__annotations__": 3, "x0e__kwdefaults__": 3, "x0c__defaults__": 3, "n__module__": 3, "x16": 3, "x07__doc__": 3, "x0b__closure__": 3, "x17_cloudpickle_submodul": 3, "x0b__globals__": 3, "x94u": 3, "x86r0": 3, "As": [3, 4, 5], "you": [3, 4, 5, 6, 7, 8, 10], "could": [3, 4, 5, 6], "see": [3, 4, 5, 8, 10], "inform": [3, 4, 6, 8], "about": [3, 4, 5, 8], "an": [3, 5, 6, 7, 8], "insepar": 3, "part": [3, 7, 8], "have": [3, 4, 5, 6, 7, 8], "sinc": [3, 4, 8], "callabl": [3, 4], "object": [3, 4, 8], "syntax": [3, 4, 6], "out": [3, 4, 5, 8, 10], "9": [3, 4, 5], "runtim": [3, 4, 5, 6, 7, 8], "none": [3, 4, 5, 6, 7, 8], "error": [3, 4, 5, 6, 7, 8], "fals": [3, 4, 5, 6, 7, 8], "wa": 3, "right": [3, 10], "awai": 3, "access": [3, 5], "later": [3, 4, 5, 7], "more": [3, 4, 5, 6, 7, 8, 10], "than": [3, 4, 6, 8], "just": [3, 4, 7], "so": [3, 4, 5, 6, 7, 8], "want": [3, 4, 5, 6, 7], "get": [3, 4, 5], "And": [3, 4], "option": [3, 4], "argument": [3, 4], "return_input": [3, 4], "true": [3, 4, 6, 7, 8], "note": [3, 4, 6, 7, 8], "default": [3, 4, 6], "alwai": [3, 4, 6, 7], "wai": [3, 4, 5, 8], "do": [3, 4, 6, 7, 8], "annot": [3, 7, 8], "anoth": [3, 5], "start": [3, 4, 5, 8], "ty": [3, 7, 8], "add_var_an": 3, "sum_a_b": 3, "int": [3, 7, 8], "task1a": 3, "might": [3, 4], "veri": [3, 6, 7], "when": [3, 4, 5, 6, 7], "modf_an": 3, "fraction": 3, "integ": [3, 4, 6], "math": 3, "modf": 3, "task2": [3, 4], "0": [3, 4, 5, 6, 7, 8], "second": [3, 4, 5, 7], "requir": [3, 6, 10], "task2a": 3, "order": [3, 4, 5, 6, 8], "don": [3, 6, 7], "provid": [3, 4, 6, 10], "task3": [3, 4], "If": [3, 4, 5, 6, 7, 10], "attr": [3, 6], "noth": [3, 8], "task3a": 3, "librari": [3, 6], "try": [3, 4, 6, 7, 8], "rais": [3, 4, 6], "typeerror": 3, "traceback": [3, 4], "most": [3, 4, 6], "recent": [3, 4], "call": [3, 4, 5], "last": [3, 4, 6], "cell": [3, 4], "In": [3, 4, 5, 6, 7, 8], "16": [3, 4, 8], "line": [3, 4], "file": [3, 4, 6, 7, 8], "usr": [3, 4], "share": [3, 4], "miniconda": [3, 4], "env": [3, 4], "tutori": [3, 4, 7, 8, 10], "lib": [3, 4], "python3": [3, 4], "11": [3, 4], "site": [3, 4], "packag": [3, 4, 7, 8], "core": [3, 4, 5], "452": 3, "taskbas": [3, 4], "__call__": 3, "self": [3, 4], "submitt": [3, 4, 5, 6, 7, 8, 9], "plugin": [3, 4, 5, 6, 7, 8, 9], "plugin_kwarg": 3, "rerun": 3, "kwarg": [3, 4], "450": 3, "re": [3, 8], "sub": [3, 4, 5, 6, 7, 8, 9], "451": 3, "els": 3, "without": [3, 4, 7], "state": [3, 10], "_run": 3, "453": 3, "510": 3, "508": 3, "509": 3, "monitor": 3, "_run_task": 3, "511": 3, "_collect_output": 3, "output_dir": [3, 7], "512": 3, "except": [3, 4, 6], "192": 3, "190": 3, "del": 3, "191": 3, "output_": 3, "cp": 3, "load": [3, 7, 8], "193": 3, "output_nam": [3, 4], "el": [3, 4], "output_spec": [3, 6], "field": [3, 6], "194": 3, "2": [3, 4, 5, 7], "6": [3, 4], "unsupport": 3, "operand": 3, "_noth": 3, "after": [3, 6], "where": [3, 6, 7, 8], "posixpath": [3, 6], "tmp8ijysl20": 3, "functiontask_a53ec5d131aa90b539e4f40f6a3d608ab3358aeef5c3958d15efd7fbbbb3ac99": 3, "find": [3, 6], "_result": 3, "pklz": 3, "os": [3, 7, 8], "listdir": 3, "_task": 3, "But": [3, 4, 7], "path": [3, 7, 8], "store": [3, 4, 8], "node": 3, "instead": [3, 4, 5, 7], "temporari": 3, "specif": [3, 4, 6, 8], "subdirectori": 3, "task4": [3, 4], "tempfil": 3, "mkdtemp": 3, "pathlib": [3, 7, 8], "cache_dir_tmp": 3, "tmps0j6k1tw": 3, "now": [3, 4, 5, 6, 7, 8], "pass": [3, 5, 6], "thi": [3, 4, 5, 6, 7, 8, 10], "cache_dir": 3, "To": [3, 6, 7], "observ": 3, "time": [3, 4, 5, 8], "sleep": [3, 4], "5s": [3, 7], "add_var_wait": 3, "first": [3, 4, 5, 6, 10], "take": [3, 4, 5, 7, 8], "around": [3, 4], "10": [3, 4, 7, 8], "our": [3, 4, 5, 6, 7, 8], "class": [3, 5, 6, 9], "checksum": [3, 6], "functiontask_44e56422e3f3d06f87f4bc9c4f0c1a79bd1a8113bd2374ce0e6bc7eed285b8ca": 3, "what": [3, 5, 7], "happen": [3, 7], "defin": [3, 4, 6], "ident": 3, "again": [3, 4, 5], "same": [3, 4, 5, 6, 7, 8], "task4a": 3, "readi": [3, 5], "avail": [3, 4, 6, 7], "onli": [3, 4, 6, 7, 8], "list": [3, 5, 6, 7, 8], "other": [3, 5, 7], "locat": 3, "previou": [3, 4, 5, 7, 8], "work": [3, 4, 6], "cache_loc": 3, "cache_dir_tmp_new": 3, "task4b": 3, "quickli": 3, "exist": 3, "regardless": 3, "alreadi": [3, 4], "sever": [3, 6], "new": [3, 5, 6, 7], "task4c": 3, "updat": [3, 4], "differ": [3, 4, 6, 8], "tmp5dih4clv": 3, "functiontask_ddcb8cfddf9a0fb53d6a5c6664a47145ab4953954f730a073fabdb9a9db7183a": 3, "becaus": 3, "chang": [3, 4], "either": 3, "number": [3, 4, 5, 7, 8], "mean": [3, 5, 8], "std": 3, "standard": 3, "deviat": 3, "mean_dev": 3, "my_list": 3, "statist": 3, "st": 3, "stdev": 3, "my_task": 3, "write": [3, 6, 8], "your": [3, 4, 6, 7, 8], "solut": [3, 6], "here": [3, 6, 7, 8], "modul": 3, "record": 3, "variou": 3, "includ": [3, 8], "audit_flag": 3, "messeng": 3, "auditflag": 3, "resourc": 3, "allow": [3, 6, 8], "usag": 3, "while": [3, 8], "prov": 3, "util": 3, "printmesseng": 3, "task5": [3, 4], "rss_peak_gb": 3, "0845451357421875": 3, "vms_peak_gb": 3, "824581146484375": 3, "cpu_peak_perc": 3, "116": 3, "8": [3, 4], "One": [3, 5], "turn": 3, "both": [3, 4], "flag": [3, 6], "all": [3, 4, 5, 6, 7, 8], "messag": 3, "termin": 3, "id": 3, "e9d78f7d9185430bb1682bb97b4eca6a": 3, "context": 3, "http": [3, 7, 8], "raw": [3, 8], "githubusercont": 3, "com": [3, 7, 8], "master": 3, "schema": 3, "jsonld": 3, "uid": 3, "b3199d9283de41b98d806f169279e9b9": 3, "startedattim": 3, "2023": 3, "16t17": 3, "25": [3, 4, 5], "07": 3, "281871": 3, "executedbi": 3, "1954132e391a412eb145bf5b48ddbe15": 3, "59af3193222e4df9945dba12b176a072": 3, "label": 3, "null": 3, "486831": 3, "associatedwith": 3, "f84b125412414d41b0ef98c5d7852d1f": 3, "00d2db0b3033432eb4ae8d5e89b79d5b": 3, "487468": 3, "wasstartedbi": 3, "781b75544b14434c850800e8468d667f": 3, "endedattim": 3, "531611": 3, "wasendedbi": 3, "083b8a34eeb541b49b20670273def216": 3, "121": 3, "ffbfd2a569c64ada8dc122adb75449a8": 3, "wasgeneratedbi": 3, "bba42b782b4a45d9bbf0f93fe6229436": 3, "entity_gener": 3, "hadact": 3, "b7347c8bd26949d2ab0b73e9ac4ff126": 3, "531900": 3, "singl": [4, 5, 8], "set": [4, 7], "iter": 4, "simpl": [4, 6, 7, 8], "add_two": [4, 5], "x": [4, 5, 8], "method": [4, 5], "one": [4, 5, 6, 7, 8], "i": [4, 5, 7, 8], "e": [4, 5, 6, 7], "0x7fa2e99a2c90": 4, "0x7fa2e46a5f50": 4, "been": 4, "add": [4, 5, 7, 8], "name": [4, 5, 6, 7, 8], "result": [4, 5, 6, 7, 8], "togeth": [4, 7, 8], "addit": [4, 5, 6], "val": 4, "indic": 4, "ind": 4, "For": [4, 7, 8], "prepar": 4, "each": [4, 5, 7, 10], "simpli": [4, 5], "repres": 4, "follow": [4, 5, 6, 7, 8, 9, 10], "figur": 4, "depend": 4, "applic": 4, "exampl": [4, 5, 6, 8], "over": 4, "12": 4, "13": [4, 8], "three": [4, 5, 7, 8], "element": [4, 5, 6], "assum": 4, "100": [4, 8], "four": [4, 5, 8], "situat": 4, "parenthes": 4, "102": 4, "expect": [4, 5, 8], "bracket": 4, "101": 4, "howev": 4, "overwrit": 4, "realli": 4, "intend": 4, "562": 4, "cont_dim": 4, "560": 4, "user": 4, "561": 4, "563": 4, "564": 4, "565": 4, "566": 4, "567": 4, "vel": 4, "item": [4, 8], "length": [4, 8], "doesn": 4, "limit": [4, 8], "y": [4, 5], "compon": 4, "vector": 4, "calcul": 4, "possibl": 4, "sum": [4, 5, 8], "kept": 4, "correspond": [4, 8], "x1": 4, "y1": 4, "x2": 4, "y2": 4, "add_vector": 4, "add_vect": 4, "20": [4, 5], "30": 4, "21": 4, "40": 4, "31": 4, "22": 4, "50": 4, "32": 4, "six": [4, 10], "vector1": 4, "vector2": 4, "modifi": [4, 5, 6, 7], "ad": [4, 6], "all_result": 4, "n": [4, 5, 6, 8], "group": [4, 7, 8], "task6": 4, "still": [4, 7], "task7": 4, "even": 4, "moment": 4, "lst": 4, "len": [4, 5, 7, 8], "task8": 4, "33": 4, "sai": 4, "squar": 4, "cube": 4, "separ": [4, 8], "its": [4, 5, 8], "task_ex1": 4, "27": 4, "64": 4, "125": 4, "squares_list": 4, "cubes_list": 4, "didn": 4, "talk": 4, "add_two_sleep": 4, "task9": 4, "t0": 4, "total": 4, "4425039291381836": 4, "machin": 4, "below": 4, "1s": 4, "clearli": 4, "automat": 4, "worker": [4, 10], "cf": [4, 5, 6, 7, 8, 9], "concurr": [4, 9], "futur": [4, 9], "processpoolexecutor": 4, "task10": 4, "4492270946502686": 4, "task11": 4, "457704544067383": 4, "runnabl": 4, "task12": 4, "4250478744506836": 4, "similar": [4, 6], "processor": 4, "max_work": 4, "n_proc": [4, 7, 8], "how": [4, 5, 8], "task13": 4, "4416351318359375": 4, "significantli": 4, "least": 4, "tasks9": 4, "took": [4, 5], "2s": 4, "less": [4, 8], "mult_var": 5, "pipelin": 5, "arbitrari": 5, "treat": 5, "input": [5, 7, 8], "input_spec": [5, 6, 7, 8], "wf1": 5, "taken": 5, "lazi": 5, "lzin": [5, 7, 8], "0x7f9fb0e28b10": 5, "0x7f9facad24d0": 5, "would": [5, 8], "output": [5, 7, 8], "wf": [5, 7, 8, 9], "set_output": [5, 7, 8], "lzout": [5, 7, 8], "dictionari": 5, "tupl": 5, "think": [5, 7], "produc": 5, "mani": 5, "variabl": 5, "wf2": 5, "out_": 5, "out_p": 5, "had": 5, "show": 5, "were": 5, "concept": 5, "wf3": 5, "lf": 5, "look": [5, 6, 7], "like": [5, 6, 8], "wf4": 5, "mult": 5, "45": 5, "previous": 5, "wf2a": 5, "wf5": 5, "exactli": 5, "insid": 5, "wf6": 5, "plitter": 5, "wf_out": 5, "175": 5, "receiv": 5, "behind": 5, "scene": 5, "expand": 5, "complic": 5, "wf7": 5, "28": 5, "63": 5, "did": [5, 8], "present": [5, 8], "x_list": 5, "wf8": 5, "own": [5, 6, 8], "out_m": 5, "49": 5, "pwd": 6, "cmd": 6, "shelli": 6, "cmdline": 6, "return_cod": 6, "stdout": 6, "tmpp_05tydi": 6, "shellcommandtask_50215383834238ce614b4428a1e9943ddff15f8580a384a0a57b34c760864b3a": 6, "stderr": 6, "everyth": 6, "goe": 6, "well": 6, "point": 6, "directori": [6, 7, 8], "empti": 6, "string": 6, "longer": 6, "echo": 6, "hail": 6, "cmndline": 6, "rewritten": 6, "specinfo": 6, "my_input_spec": 6, "spec": [6, 7, 8], "text": 6, "ib": 6, "str": [6, 7, 8], "metadata": 6, "posit": [6, 8], "argstr": 6, "help_str": 6, "mandatori": 6, "base": [6, 7], "shellspec": 6, "notic": 6, "valid": 6, "attribut": 6, "full": 6, "found": 6, "allowed_valu": 6, "output_field_nam": 6, "copyfil": 6, "separate_ext": 6, "container_path": 6, "xor": 6, "output_file_templ": 6, "among": 6, "sring": 6, "descript": 6, "grater": 6, "rel": 6, "g": [6, 7], "o": 6, "need": [6, 7, 8], "bool": 6, "complet": 6, "document": 6, "suport": 6, "perhap": 6, "simplest": 6, "my_input_spec_short": 6, "cmd_exec": 6, "hello": 6, "someth": 6, "my_output_spec": 6, "newfil": 6, "newfile_tmp": 6, "txt": [6, 7], "shelloutspec": 6, "touch": 6, "tmpih5_o4nv": 6, "shellcommandtask_76e73f8b52e6d5d1920bda2a22e309fb4fd89cec14e6ab25755e2caf66cf0294": 6, "newfile_1": 6, "newfile_2": 6, "out1": 6, "NOT": 6, "IF": 6, "IS": 6, "fail": 6, "It": [6, 7, 10], "binder": [6, 10], "imag": [6, 7, 8], "whoami": 6, "docki": 6, "busybox": 6, "root": 6, "unabl": 6, "latest": 6, "local": [6, 8, 9, 10], "nlatest": 6, "pull": 6, "n3f4d90098f5b": 6, "fs": 6, "layer": 6, "verifi": 6, "download": [6, 7, 10], "ndigest": 6, "sha256": 6, "3fbc632167424a6d997e74f52b878d7cc478225cffac6bc977eedfe51c7f4e79": 6, "nstatu": 6, "newer": 6, "splitter": [6, 7, 8], "ubuntu": 6, "naece8493d397": 6, "2b7412e6465c3c7fc5bb21d3e6f1917c167358449fecac8176c6e496e5c1f05f": 6, "runner": [6, 7], "through": [7, 10], "linear": 7, "analysi": [7, 8, 10], "perform": [7, 8], "subject": 7, "up": 7, "warn": [7, 8], "sy": [7, 8], "warnopt": [7, 8], "simplefilt": [7, 8], "ignor": [7, 8], "panda": [7, 8], "pd": [7, 8], "scipi": [7, 8], "stat": [7, 8], "norm": [7, 8], "nibabel": [7, 8], "nib": [7, 8], "fetch_openneuro_dataset_index": 7, "fetch_openneuro_dataset": 7, "select_from_index": 7, "interfac": [7, 8], "get_design_from_fslmat": 7, "first_level": [7, 8], "first_level_from_bid": 7, "get_clusters_t": 7, "make_glm_report": 7, "plot_glass_brain": [7, 8], "plot_img_comparison": 7, "plot_contrast_matrix": 7, "pydra_tutorial_dir": [7, 8], "dirnam": [7, 8], "getcwd": [7, 8], "workflow_dir": [7, 8], "workflow_out_dir": [7, 8], "6_glm": 7, "exit": [7, 8], "makedir": [7, 8], "exist_ok": [7, 8], "section": [7, 8], "convert": 7, "major": 7, "step": [7, 8], "recommand": 7, "put": [7, 8], "those": [7, 8, 10], "logic": 7, "relat": 7, "keep": 7, "mind": 7, "adjac": 7, "decid": [7, 10], "impact": 7, "index": [7, 8], "exclus": 7, "pattern": 7, "data": [7, 10], "n_subject": 7, "given": 7, "analyz": 7, "remov": 7, "1": 7, "forget": 7, "exclusion_pattern": 7, "data_dir": 7, "get_openneuro_dataset": 7, "_": 7, "url": 7, "exclusion_filt": 7, "task_label": 7, "space_label": 7, "desir": 7, "deriv": [7, 8], "fmriprep": [7, 8], "case": [7, 8], "Then": 7, "event": [7, 8], "confound": [7, 8], "regressor": 7, "infer": 7, "tsv": [7, 8], "derivatives_fold": 7, "smoothing_fwhm": [7, 8], "float": [7, 8], "img": [7, 8], "get_info_from_bid": 7, "models_run_img": 7, "models_ev": 7, "models_confound": 7, "dataset_path": 7, "subject_label": 7, "doe": 7, "read": [7, 8], "mat": 7, "renam": 7, "column": [7, 8], "csv": [7, 8], "directli": [7, 8], "word": 7, "datafram": [7, 8], "worth": 7, "dm_path": [7, 8], "get_designmatrix": 7, "fsl_design_matrix_path": 7, "join": [7, 8], "stopsign": 7, "feat": 7, "design_matrix": [7, 8], "column_nam": 7, "design_column": 7, "cond_": 7, "02d": [7, 8], "rang": [7, 8], "stopsuccess": 7, "designmatrix": 7, "to_csv": [7, 8], "comput": 7, "z_map": 7, "masker": 7, "futher": 7, "z_map_path": 7, "model_fit": 7, "read_csv": [7, 8], "design_matric": [7, 8], "compute_contrast": [7, 8], "firstlevel_z_map": 7, "nii": [7, 8], "gz": [7, 8], "to_filenam": [7, 8], "masker_path": 7, "firstlevel_mask": 7, "masker_": [7, 8], "public": 7, "summari": 7, "output_fil": [7, 8], "cluster_t": 7, "stat_img": [7, 8], "df": 7, "stat_threshold": 7, "isf": [7, 8], "001": [7, 8], "cluster_threshold": [7, 8], "glm_report": 7, "html": 7, "save_as_html": 7, "some": 7, "displai": [7, 8], "compar": 7, "seper": 7, "sens": 7, "them": 7, "repeatedli": 7, "output_file1": 7, "output_file2": 7, "output_file3": 7, "output_file4": 7, "nilearn_z_map": 7, "jpg": [7, 8], "colorbar": [7, 8], "threshold": 7, "titl": [7, 8], "unc": 7, "p": [7, 8], "plot_ab": 7, "display_mod": 7, "ortho": 7, "fsl_z_map": 7, "zstat12": 7, "ref_label": 7, "src_label": 7, "old": 7, "0000": 7, "png": 7, "nilearn_fsl_comp": 7, "firstlevel_contrast": [7, 8], "choic": [7, 8], "close": 7, "along": 7, "larger": 7, "approach": [7, 8], "good": 7, "practic": 7, "especi": 7, "larg": 7, "divid": 7, "firstlevel": [7, 8], "whatev": 7, "prefer": 7, "initi": [7, 8], "wf_firstlevel": [7, 8], "mni152nlin2009casym": 7, "l1estim": 7, "abov": 7, "firstlevel_glm": 7, "phenotyp": 7, "mriqc": 7, "parameter_plot": 7, "physio_plot": 7, "space": [7, 8], "fsaverag": 7, "t1w": 7, "dwi": 7, "beh": 7, "bart": 7, "rest": 7, "scap": 7, "output1": 7, "output2": 7, "output3": 7, "output4": 7, "home": 7, "nilearn_data": 7, "ds000030": 7, "ds000030_r1": 7, "uncompress": 7, "osf": 7, "io": 7, "86xj7": 7, "done": [7, 8], "min": 7, "s3": 7, "amazonaw": 7, "readm": 7, "dataset_descript": 7, "json": 7, "10159": 7, "anat": 7, "10159_t1w_brainmask": 7, "10159_t1w_dtissu": 7, "10159_t1w_inflat": 7, "l": 7, "surf": 7, "gii": 7, "r": 7, "10159_t1w_midthick": 7, "10159_t1w_pial": 7, "10159_t1w_preproc": 7, "10159_t1w_smoothwm": 7, "10159_t1w_space": 7, "mni152nlin2009casym_brainmask": 7, "mni152nlin2009casym_class": 7, "csf_probtissu": 7, "gm_probtissu": 7, "wm_probtissu": 7, "mni152nlin2009casym_preproc": 7, "mni152nlin2009casym_warp": 7, "h5": 7, "69148672": 7, "102374780": 7, "byte": 7, "67": 7, "remain": 7, "func": [7, 8], "10159_task": 7, "stopsignal_bold_confound": 7, "stopsignal_bold_spac": 7, "52338688": 7, "120927302": 7, "43": 7, "3s": 7, "91856896": 7, "76": 7, "6s": 7, "css": 7, "1r": 7, "flirt": 7, "bg": 7, "logo": 7, "big": 7, "gif": 7, "maco": 7, "snapshot": 7, "tiff": 7, "fslstart": 7, "fugu": 7, "tick": 7, "vert2": 7, "ramp": 7, "absbrainthresh": 7, "custom_timing_fil": 7, "ev1": 7, "ev10": 7, "ev11": 7, "ev12": 7, "ev13": 7, "ev14": 7, "ev15": 7, "ev16": 7, "ev17": 7, "ev18": 7, "ev19": 7, "ev2": 7, "ev3": 7, "ev4": 7, "ev5": 7, "ev6": 7, "ev7": 7, "ev8": 7, "ev9": 7, "con": 7, "frf": 7, "fsf": 7, "ppm": 7, "trg": 7, "design_cov": 7, "example_func": 7, "filtered_func_data": 7, "log": [7, 8], "feat0": 7, "feat0_init": 7, "e60127": 7, "o60127": 7, "feat1": 7, "feat1a_init": 7, "feat2_pr": 7, "e60564": 7, "o60564": 7, "feat3_film": 7, "e61431": 7, "o61431": 7, "feat3_stat": 7, "feat4_post": 7, "e120148": 7, "o120148": 7, "feat5_stop": 7, "e134343": 7, "o134343": 7, "feat9": 7, "mask": [7, 8], "mean_func": 7, "report_log": 7, "report_poststat": 7, "report_prestat": 7, "report_reg": 7, "report_stat": 7, "cope1": 7, "cope10": 7, "cope11": 7, "cope12": 7, "cope13": 7, "cope14": 7, "cope15": 7, "cope16": 7, "cope17": 7, "cope18": 7, "cope19": 7, "cope2": 7, "cope20": 7, "cope21": 7, "cope22": 7, "cope23": 7, "cope24": 7, "cope3": 7, "cope4": 7, "cope5": 7, "cope6": 7, "cope7": 7, "cope8": 7, "cope9": 7, "dof": 7, "logfil": 7, "pe1": 7, "pe10": 7, "pe11": 7, "pe12": 7, "pe13": 7, "pe14": 7, "pe15": 7, "pe16": 7, "pe17": 7, "pe18": 7, "pe19": 7, "pe2": 7, "pe20": 7, "pe21": 7, "pe22": 7, "pe23": 7, "pe24": 7, "pe25": 7, "pe26": 7, "pe27": 7, "pe28": 7, "pe3": 7, "pe4": 7, "pe5": 7, "pe6": 7, "pe7": 7, "pe8": 7, "pe9": 7, "res4d": 7, "sigmasquar": 7, "smooth": 7, "threshac1": 7, "tstat1": 7, "tstat10": 7, "tstat11": 7, "tstat12": 7, "tstat13": 7, "tstat14": 7, "tstat15": 7, "tstat16": 7, "tstat17": 7, "tstat18": 7, "tstat19": 7, "tstat2": 7, "tstat20": 7, "tstat21": 7, "tstat22": 7, "tstat23": 7, "tstat24": 7, "tstat3": 7, "tstat4": 7, "tstat5": 7, "tstat6": 7, "tstat7": 7, "tstat8": 7, "tstat9": 7, "varcope1": 7, "varcope10": 7, "varcope11": 7, "varcope12": 7, "varcope13": 7, "varcope14": 7, "varcope15": 7, "varcope16": 7, "varcope17": 7, "varcope18": 7, "varcope19": 7, "varcope2": 7, "varcope20": 7, "varcope21": 7, "varcope22": 7, "varcope23": 7, "varcope24": 7, "varcope3": 7, "varcope4": 7, "varcope5": 7, "varcope6": 7, "varcope7": 7, "varcope8": 7, "varcope9": 7, "zstat1": 7, "zstat10": 7, "zstat11": 7, "zstat13": 7, "zstat14": 7, "zstat15": 7, "zstat16": 7, "zstat17": 7, "zstat18": 7, "zstat19": 7, "zstat2": 7, "zstat20": 7, "zstat21": 7, "zstat22": 7, "zstat23": 7, "zstat24": 7, "zstat3": 7, "zstat4": 7, "zstat5": 7, "zstat6": 7, "zstat7": 7, "zstat8": 7, "zstat9": 7, "e3328": 7, "o3328": 7, "e4486": 7, "o4486": 7, "e8788": 7, "o8788": 7, "e22680": 7, "o22680": 7, "e58254": 7, "o58254": 7, "taskswitch": 7, "ev20": 7, "ev21": 7, "ev22": 7, "ev23": 7, "ev24": 7, "ev25": 7, "ev26": 7, "e182384": 7, "o182384": 7, "e183363": 7, "o183363": 7, "e183962": 7, "o183962": 7, "e194150": 7, "o194150": 7, "e52571": 7, "o52571": 7, "cope25": 7, "cope26": 7, "cope27": 7, "cope28": 7, "cope29": 7, "cope30": 7, "cope31": 7, "cope32": 7, "cope33": 7, "cope34": 7, "cope35": 7, "cope36": 7, "cope37": 7, "cope38": 7, "cope39": 7, "cope40": 7, "cope41": 7, "cope42": 7, "cope43": 7, "cope44": 7, "cope45": 7, "cope46": 7, "cope47": 7, "cope48": 7, "pe29": 7, "pe30": 7, "pe31": 7, "pe32": 7, "pe33": 7, "pe34": 7, "pe35": 7, "pe36": 7, "pe37": 7, "pe38": 7, "pe39": 7, "pe40": 7, "pe41": 7, "pe42": 7, "tstat25": 7, "tstat26": 7, "tstat27": 7, "tstat28": 7, "tstat29": 7, "tstat30": 7, "tstat31": 7, "tstat32": 7, "tstat33": 7, "tstat34": 7, "tstat35": 7, "tstat36": 7, "tstat37": 7, "tstat38": 7, "tstat39": 7, "tstat40": 7, "tstat41": 7, "tstat42": 7, "tstat43": 7, "tstat44": 7, "tstat45": 7, "tstat46": 7, "tstat47": 7, "tstat48": 7, "varcope25": 7, "varcope26": 7, "varcope27": 7, "varcope28": 7, "varcope29": 7, "varcope30": 7, "varcope31": 7, "varcope32": 7, "varcope33": 7, "varcope34": 7, "varcope35": 7, "varcope36": 7, "varcope37": 7, "varcope38": 7, "varcope39": 7, "varcope40": 7, "varcope41": 7, "varcope42": 7, "varcope43": 7, "varcope44": 7, "varcope45": 7, "varcope46": 7, "varcope47": 7, "varcope48": 7, "zstat25": 7, "zstat26": 7, "zstat27": 7, "zstat28": 7, "zstat29": 7, "zstat30": 7, "zstat31": 7, "zstat32": 7, "zstat33": 7, "zstat34": 7, "zstat35": 7, "zstat36": 7, "zstat37": 7, "zstat38": 7, "zstat39": 7, "zstat40": 7, "zstat41": 7, "zstat42": 7, "zstat43": 7, "zstat44": 7, "zstat45": 7, "zstat46": 7, "zstat47": 7, "zstat48": 7, "particip": 7, "10159_t1w": 7, "stopsignal_bold": 7, "stopsignal_ev": 7, "bht_bold": 7, "pamenc_bold": 7, "pamret_bold": 7, "arriv": 7, "yai": 7, "made": 7, "got": 7, "ls": 7, "ipython": [7, 8], "filenam": [7, 8], "demonstr": 8, "balloon": 8, "analog": 8, "risk": 8, "basic": 8, "dataset": 8, "scan": 8, "tr": 8, "num": 8, "300": 8, "glob": 8, "datetim": 8, "random": 8, "multiinputfil": 8, "multioutputfil": 8, "datalad": 8, "dl": 8, "numpi": 8, "np": 8, "load_confounds_strategi": 8, "load_img": 8, "get_data": 8, "math_img": 8, "threshold_img": 8, "make_first_level_design_matrix": 8, "firstlevelmodel": 8, "second_level": 8, "secondlevelmodel": 8, "non_parametric_infer": 8, "compute_fixed_effect": 8, "plot_stat_map": 8, "7_glm": 8, "folder": 8, "often": 8, "preprocess": 8, "fmriprep_path": 8, "rawdata_path": 8, "raw_data": 8, "datapath": 8, "fmriprep_url": 8, "github": 8, "openneuroderiv": 8, "ds000001": 8, "git": 8, "rawdata_url": 8, "openneurodataset": 8, "sourc": 8, "By": 8, "symlink": 8, "actual": 8, "event_info": 8, "bold": 8, "mni152nlin2009casym_r": 8, "2_desc": 8, "preproc_bold": 8, "brain_mask": 8, "desc": 8, "confounds_timeseri": 8, "implicitli": 8, "subj_id": 8, "subj_ev": 8, "subj_img": 8, "subj_mask": 8, "get_subjdata": 8, "ndownload": 8, "sort": 8, "subj_confound": 8, "conduct": 8, "averag": 8, "across": 8, "model": 8, "m": 8, "row": 8, "condit": 8, "n_scan": 8, "hrf_model": 8, "run_id": 8, "get_firstlevel_dm": 8, "nget": 8, "run_img": 8, "run_ev": 8, "sep": 8, "fillna": 8, "onset": 8, "durat": 8, "trial_typ": 8, "denoise_strategi": 8, "frame_tim": 8, "arang": 8, "add_reg": 8, "make": 8, "sure": [8, 10], "matric": 8, "block": 8, "39": 8, "34": 8, "drift": 8, "constant": 8, "assert": 8, "shape": 8, "52": 8, "wrong": 8, "alphabet": 8, "reindex": 8, "axi": 8, "s_run": 8, "s_designmatrix": 8, "dict": 8, "set_contrast": 8, "nset": 8, "contrast_matrix": 8, "ey": 8, "basic_contrast": 8, "enumer": 8, "pump": 8, "control": 8, "pumps_demean": 8, "control_pumps_demean": 8, "baselin": 8, "cash": 8, "cash_demean": 8, "explod": 8, "explode_demean": 8, "effect_size_path_dict": 8, "effect_variance_path_dict": 8, "firstlevel_estim": 8, "nstart": 8, "estim": 8, "subsampl": 8, "memori": 8, "img_data": 8, "new_img": 8, "nifti1imag": 8, "affin": 8, "run_mask": 8, "first_level_model": 8, "mask_img": 8, "dm": 8, "fromkei": 8, "contrast_id": 8, "contrast_v": 8, "2i": 8, "contast": 8, "session": 8, "output_typ": 8, "effect_size_path": 8, "s_contrast": 8, "s_effect_s": 8, "effect_variance_path": 8, "s_effect_varainc": 8, "effect_s": 8, "effect_vari": 8, "first_level_contrast": 8, "first_level_effect_size_list": 8, "first_level_effect_variance_list": 8, "move": 8, "effect_size_path_dict_list": 8, "effect_variance_path_dict_list": 8, "fixed_fx_contrast_path_dict": 8, "fixed_fx_variance_path_dict": 8, "fixed_fx_ttest_path_dict": 8, "get_fixed_effct": 8, "mean_mask": 8, "binar": 8, "contrast_img": 8, "img_dict": 8, "variance_img": 8, "fixed_fx_contrast": 8, "fixed_fx_vari": 8, "fixed_fx_ttest": 8, "s_fx_effect_s": 8, "variance_path": 8, "s_fx_effect_varainc": 8, "ttest_path": 8, "s_ttest_map": 8, "wf_fixed_effect": 8, "get_subj_fil": 8, "fx_effect_size_list": 8, "fx_effect_variance_list": 8, "fx_t_test_list": 8, "known": 8, "sampl": 8, "ones": 8, "intercept": 8, "n_subj": 8, "get_secondlevel_dm": 8, "t1": 8, "secondlevel": 8, "z": 8, "firstlevel_stats_list": 8, "secondlevel_mask": 8, "stat_maps_dict": 8, "secondlevel_estim": 8, "second_level_input": 8, "stats_dict": 8, "second_level_model": 8, "mask_img_": 8, "z_image_path": 8, "secondlevel_contrast": 8, "s_z_map": 8, "z_score": 8, "plot_path": 8, "secondlevel_unthresholded_contrast": 8, "s_zmap": 8, "wf_secondlevel": 8, "n_perm": 8, "second_level_designmatrix": 8, "second_level_mask": 8, "second_level_stats_map": 8, "fdr": 8, "ab": 8, "29": 8, "equival": 8, "size": 8, "voxel": 8, "thresholded_map_dict": 8, "plot_contrast_dict": 8, "stats_id": 8, "stats_val": 8, "thresholded_map": 8, "two_sid": 8, "thresholded_map_path": 8, "secondlevel_cluster_thresholded_contrast": 8, "ncluster": 8, "discoveri": 8, "rate": 8, "05": 8, "fpr": 8, "bonferroni": 8, "detail": 8, "alpha": 8, "height_control": 8, "multiple_comparison": 8, "threshold_stats_img": 8, "secondlevel_multiple_comp_corrected_contrast": 8, "nmultipl": 8, "uncorrect": 8, "parametric_test": 8, "parametr": 8, "p_val": 8, "p_valu": 8, "n_voxel": 8, "neg": 8, "logarithm": 8, "neg_log_pv": 8, "log10": 8, "minimum": 8, "format": 8, "secondlevel_paramatric_thresholded_contrast": 8, "equal": 8, "lower": 8, "probabl": 8, "90": 8, "chanc": 8, "much": 8, "conserv": 8, "fwer": 8, "nparametr": 8, "permut": 8, "nonparametric_test": 8, "nonparametr": 8, "neg_log_pvals_permuted_ols_unmask": 8, "model_intercept": 8, "two_sided_test": 8, "n_job": 8, "secondlevel_permutation_contrast": 8, "npermut": 8, "randomli": 8, "choos": 8, "twolevel_glm": 8, "17": 8, "glover": 8, "ok": 8, "10_task": 8, "balloonanalogrisktask_run": 8, "1_space": 8, "openneuro": 8, "2_space": 8, "3_space": 8, "1_desc": 8, "3_desc": 8, "15": 8, "15_task": 8, "16_task": 8, "05_task": 8, "14": 8, "14_task": 8, "nifti1": 8, "0x7f00b2e53850": 8, "0x7f00b2508810": 8, "0x7f00b250b990": 8, "0x7f00b2541290": 8, "0x7f00b2542f90": 8, "0x7f00b2543150": 8, "0x7f00b2541c10": 8, "0x7f00b25437d0": 8, "0x7f00b2540710": 8, "0x7f00b2542550": 8, "0x7f00b2543550": 8, "0x7f00b2543690": 8, "0x7f00b27ebbd0": 8, "0x7f00b2839390": 8, "0x7f00b2839dd0": 8, "0x7f00b29a9c90": 8, "0x7f00b29adb90": 8, "0x7f00b24a1490": 8, "0x7f00b24a01d0": 8, "0x7f00b24a1810": 8, "0x7f00b24a1150": 8, "0x7f00b24a2410": 8, "0x7f00b298d790": 8, "0x7f00b2730ed0": 8, "0x7f00b2731ed0": 8, "reason": 8, "surviv": 8, "ut_list": 8, "secondlevel_unthreshold": 8, "ct_list": 8, "secondlevel_cluster_threshold": 8, "mc_list": 8, "secondlevel_multiple_comp": 8, "pt_list": 8, "secondlevel_paramatr": 8, "npt_list": 8, "secondlevel_permut": 8, "per": 8, "why": 8, "moreov": 8, "due": 8, "simultan": 8, "give": 8, "linearli": 9, "serial": 9, "parallel": 9, "system": 9, "sge": 9, "executor": 9, "psij": 9, "book": 10, "walk": 10, "hand": 10, "experi": 10, "cover": 10, "topic": 10, "philosophi": 10, "bid": 10, "plai": 10, "click": 10, "button": 10, "top": 10, "chapter": 10, "necessari": 10, "intro": 10, "glm": 10, "nilearn": 10, "team": 10, "cite": 10}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"cite": 0, "pydra": [0, 2, 9, 10], "team": 1, "intro": 2, "comput": [2, 8], "object": [2, 7], "task": [2, 4, 5, 7], "worker": [2, 9], "functiontask": 3, "custom": [3, 6], "output": [3, 4, 6], "name": 3, "set": [3, 5, 8], "input": [3, 4, 6], "directori": 3, "cach": 3, "result": 3, "exercis": [3, 4, 6, 7, 8], "1": [3, 4, 6, 8, 9], "us": [3, 6], "audit": 3, "state": 4, "multipl": [4, 8], "split": 4, "scalar": 4, "splitter": [4, 5], "outer": 4, "combin": [4, 5], "list": 4, "an": 4, "parallel": 4, "execut": 4, "workflow": [5, 7, 8], "connect": 5, "node": 5, "ad": 5, "shellcommandtask": 6, "command": 6, "argument": [6, 7], "arg": 6, "dockertask": 6, "exercise2": 6, "container_info": 6, "first": [7, 8], "level": [7, 8], "glm": [7, 8], "from": [7, 8], "nilearn": [7, 8], "prepar": [7, 8], "creat": [7, 8], "fetch": 7, "openneuro": 7, "bid": 7, "dataset": 7, "obtain": 7, "firstlevelmodel": 7, "automat": 7, "fit": [7, 8], "get": [7, 8], "design": [7, 8], "matrix": [7, 8], "model": [7, 10], "cluster": [7, 8], "tabl": 7, "report": 7, "make": 7, "plot": [7, 8], "The": [7, 8], "overach": 7, "run": [7, 8], "visual": 7, "examin": 7, "folder": 7, "structur": 7, "figur": 7, "contrast": [7, 8], "z": 7, "map": 7, "fsl": 7, "comparison": [7, 8], "two": 8, "download": 8, "data": 8, "each": 8, "subject": 8, "up": 8, "fix": 8, "effect": 8, "second": 8, "statist": 8, "test": 8, "threshold": 8, "without": 8, "paramatr": 8, "non": 8, "ultim": 8, "let": 8, "s": 8, "unthreshold": 8, "nonparamatr": 8, "2": [8, 9], "serialwork": 9, "concurrentfutureswork": 9, "3": 9, "slurmwork": 9, "4": 9, "daskwork": 9, "5": 9, "sgework": 9, "6": 9, "psijwork": 9, "welcom": 10, "basic": 10, "concept": 10, "gener": 10, "linear": 10, "about": 10}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})