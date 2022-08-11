---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import nest_asyncio
nest_asyncio.apply()
```

```{code-cell} ipython3
import os, glob
import datetime
import pydra
from pydra import Workflow
from pydra.engine.specs import File, MultiInputFile, MultiOutputFile
import typing as ty
from pathlib import Path

# get current directory
pydra_tutorial_dir = os.path.dirname(os.getcwd())

# set up output directory
workflow_dir = Path(pydra_tutorial_dir) / 'outputs'
workflow_out_dir = workflow_dir / '7_glm' / 'test_pydra'

# create the output directory if not exit
os.makedirs(workflow_out_dir, exist_ok=True)
```

```{code-cell} ipython3
event_list = glob.glob(os.path.join(rawdata_path, '*', 'func', '*events.tsv'))
import datalad.api as dl
event_list.sort()
for i in event_list:
    dl.get(i)
```

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'rawdata_url': str,
        'fmriprep_url': str,
        'return': {'event_list': MultiOutputFile, 
                   'img_list': MultiOutputFile, 
                   'mask_list': MultiOutputFile, 
                  },
    }
)
def get_data(rawdata_url, fmriprep_url):
    print("Download data...")
    t1 = datetime.datetime.now()
    print(t1)
    import datalad.api as dl
    fmriprep_path = workflow_dir / '7_glm'/ 'data'
    rawdata_path = workflow_dir / '7_glm' / 'raw_data'
    
    # Install datasets to specific datapaths
    dl.install(source=rawdata_url, path=fmriprep_path)
    dl.install(source=fmriprep_url, path=rawdata_path)
    
    # get events.tsv list
    event_list = glob.glob(os.path.join(rawdata_path, '*', 'func', '*events.tsv'))
    event_list.sort()
    for i in event_list:
        dl.get(i, dataset=rawdata_path)
    # get img list
    img_list = glob.glob(os.path.join(fmriprep_path, '*', 'func', '*space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'))
    img_list.sort()
    for i in img_list:
        dl.get(i, dataset=fmriprep_path)
    
     # get img list
    mask_list = glob.glob(os.path.join(fmriprep_path, '*', 'func', '*space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz'))
    mask_list.sort()
    for i in mask_list:
        dl.get(i, dataset=fmriprep_path)

    t2 = datetime.datetime.now()
    print(t2-t1)
    return event_list, img_list, mask_list
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: []
---
task1 = get_data(
    rawdata_url = 'https://github.com/OpenNeuroDerivatives/ds000001-fmriprep.git',
    fmriprep_url = 'https://github.com/OpenNeuroDatasets/ds000001.git')
task1()
```

```{code-cell} ipython3
:tags: []

result1 = task1.result()
```

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'subj_id': int,
        'n_run': int,
        'event_list': list, 
        'img_list': list, 
        'mask_list': list,
        'return': {'subj_id': int, 'subj_events': list, 'subj_imgs':list, 'subj_masks':list},
    }
)
def get_subj_file(subj_id, n_run, event_list, img_list, mask_list):
    t1 = datetime.datetime.now()
    print(f"Get subject-{subj_id} file...\n")
    # subj_id starts from 0
    start = subj_id*n_run
    end = (subj_id+1)*n_run
    subj_events = event_list[start:end]
    subj_imgs = img_list[start:end]
    subj_masks = mask_list[start:end]
    t2 = datetime.datetime.now()
    print(t2-t1)
    return subj_id, subj_events, subj_imgs, subj_masks
```

```{code-cell} ipython3
:tags: []

task2 = get_subj_file(subj_id=[1,2,3], n_run=3, 
                      event_list=result1.output.event_list, 
                      img_list=result1.output.img_list, 
                      mask_list=result1.output.mask_list).split('subj_id')
task2()
result2 = task2.result()
```

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'tr': float,
        'n_scans': int,
        'hrf_model': str,
        'subj_id': int,
        'subj_imgs': list,
        'subj_events':list,
        'return': {'design_matrices': list, 'dm_paths':list},
    }
)
def get_firstlevel_dm(tr, n_scans, hrf_model, subj_id, subj_imgs, subj_events):
    t1 = datetime.datetime.now()
    print("Get firstlevel GLM ...\n")
    import numpy as np
    import pandas as pd
    from nilearn.glm.first_level import make_first_level_design_matrix
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    # read event file
    events = []
    imgs = []
    for run_event in subj_events:
        event = pd.read_csv(run_event, sep='\t').fillna(0)
        event = event[['onset', 'duration', 'trial_type']]
        events.append(event)
    
    # get list of confounds directly from fmriprepped bold
    confounds = load_confounds_strategy(subj_imgs, denoise_strategy='simple')[0]
    
    frame_times = np.arange(n_scans) * tr
    design_matrices = []
    dm_paths = []
    for index, (ev, conf) in enumerate(zip(events, confounds)):
        design_matrix = make_first_level_design_matrix(frame_times, ev, 
                                                       hrf_model=hrf_model,
                                                       add_regs=conf)          
        
        # make sure all design matrices have the same length of column
        # if you have a block design, this is not needed.
        # 39 = 4(events) + 34(confounds) + 13(drift) + 1(constant)
        assert design_matrix.shape[1] == 52, "This design matrix has the wrong column number"
        # sort the column order alphabetical for contrasts
        design_matrix = design_matrix.reindex(sorted(design_matrix.columns), axis=1)
        dm_path = os.path.join(workflow_out_dir, 'sub-%s_run-%s_designmatrix.csv' % (subj_id, index+1))
        design_matrix.to_csv(dm_path, index=None)
        design_matrices.append(design_matrix)
        dm_paths.append(dm_path)
    t2 = datetime.datetime.now()
    print(t2-t1)
    return design_matrices, dm_paths
```

```{code-cell} ipython3
:tags: []

task3 = get_firstlevel_dm(tr=2.3, n_scans=300, hrf_model='glover', 
                          subj_id=result2[0].output.subj_id, 
                          subj_imgs=result2[0].output.subj_imgs, 
                          subj_events=result2[0].output.subj_events)
task3()
result3 = task3.result()
```

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'subj_id': int,
        'design_matrices': list,
        'return': {'contrasts': dict, 'contrast_plot':list},
    }
)
def set_contrast(subj_id, design_matrices):
    t1 = datetime.datetime.now()
    print(f"Set firstlevel contrast for subject-{subj_id} ...\n")
    
    import pandas as pd
    import numpy as np
    from nilearn.plotting import plot_contrast_matrix
    
    design_matrix = design_matrices[0]
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                      for i, column in enumerate(design_matrix.columns)])
    contrasts = {
        'pumps-control': basic_contrasts['pumps_demean'] - basic_contrasts['control_pumps_demean'],
        'control-pumps': -basic_contrasts['control_pumps_demean'] + basic_contrasts['pumps_demean'],
        'pumps-baseline': basic_contrasts['pumps_demean'],
        'cash-baseline': basic_contrasts['cash_demean'],
        'explode-baseline': basic_contrasts['explode_demean'],
        'effects_of_interest': np.vstack((basic_contrasts['pumps_demean'],
                                          basic_contrasts['cash_demean'],
                                          basic_contrasts['explode_demean']))
        }
    
    contrast_plot = []
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Plot Contrast % 2i out of %i: %s' % (
            index + 1, len(contrasts), contrast_id))
        contrast_plot_path = os.path.join(workflow_out_dir, 'sub-%s_firstlevel_contrast-%s.jpg' % (subj_id, contrast_id))
        plot_contrast_matrix(contrast_val, design_matrix, output_file=contrast_plot_path)
        contrast_plot.append(contrast_plot_path)
    t2 = datetime.datetime.now()
    print(t2-t1)
    return contrasts, contrast_plot
```

```{code-cell} ipython3
task4 = set_contrast(
    subj_id=result2[0].output.subj_id, 
    design_matrices=result3.output.design_matrices)

task4()
result4 = task4.result()
```

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'subj_id': int,
        'subj_imgs': MultiInputFile,
        'subj_masks': MultiInputFile,
        'smoothing_fwhm': float,
        'design_matrices': list,
        'contrasts':dict,
        'return': {'first_level_model': ty.Any, 'z_map_path_dict': dict},
    }
)
def firstlevel_estimation(subj_id, subj_imgs, subj_masks, smoothing_fwhm, design_matrices, contrasts):
    t1 = datetime.datetime.now()
    print(f"Start firstlevel estimation for subject-{subj_id} ...\n")
    
    import nibabel as nib
    from nilearn.image import math_img
    from nilearn.glm.first_level import FirstLevelModel
    
    print('Compute firstlevel mask...')
    # average mask across three runs
    mean_mask = math_img('np.mean(img, axis=-1)', img=subj_masks)
    # binarize the mean mask
    mask = math_img('img > 0', img=mean_mask)
    # fit the (fixed-effects) firstlevel model with three runs simultaneously
    first_level_model = FirstLevelModel(mask_img=mask, smoothing_fwhm=smoothing_fwhm, minimize_memory=True)
    first_level_model = first_level_model.fit(subj_imgs, design_matrices=design_matrices)
    
    print('Computing contrasts...')
    z_map_path_dict = dict.fromkeys(contrasts.keys())
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % 2i out of %i: %s' % (
            index + 1, len(contrasts), contrast_id))
        # Estimate the contasts. Note that the model implicitly computes a fixed
        # effect across the two sessions
        z_map = first_level_model.compute_contrast(
            contrast_val, output_type='z_score')

        # write the resulting stat images to file
        z_map_path = os.path.join(workflow_out_dir, 'sub-%s_contrast-%s_z_map.nii.gz' % (subj_id, contrast_id))
        z_map_path_dict[contrast_id] = z_map_path
        z_map.to_filename(z_map_path)
    
    t2 = datetime.datetime.now()
    print(t2-t1)
    return first_level_model, z_map_path_dict
```

```{code-cell} ipython3
task5 = firstlevel_estimation(
    subj_id=result2[0].output.subj_id, 
    subj_imgs=result2[0].output.subj_imgs, 
    subj_masks=result2[0].output.subj_masks, 
    smoothing_fwhm=5.0, 
    design_matrices=result3.output.design_matrices, 
    contrasts=result4.output.contrasts,)

task5()
result5 = task5.result()
```

```{code-cell} ipython3
# initiate the first-level GLM workflow
wf_firstlevel = Workflow(
    name='wf_firstlevel',
    input_spec=[
        'subj_id',
        'n_run',
        'tr',
        'n_scans',
        'hrf_model',
        'event_list', 
        'img_list', 
        'mask_list',
        'smoothing_fwhm',
        'output_dir'
    ],
)

# add task - get_subj_file
wf_firstlevel.add(
    get_subj_file(
        name = "get_subj_file",
        subj_id = wf_firstlevel.lzin.subj_id, 
        n_run = wf_firstlevel.lzin.n_run, 
        event_list = wf_firstlevel.lzin.event_list, 
        img_list = wf_firstlevel.lzin.img_list, 
        mask_list = wf_firstlevel.lzin.mask_list
    )
)

# add task - get_firstlevel_dm
wf_firstlevel.add(
    get_firstlevel_dm(
        name = "get_firstlevel_dm",
        tr = wf_firstlevel.lzin.tr, 
        n_scans = wf_firstlevel.lzin.n_scans, 
        hrf_model = wf_firstlevel.lzin.hrf_model, 
        subj_id = wf_firstlevel.get_subj_file.lzout.subj_id, 
        subj_imgs = wf_firstlevel.get_subj_file.lzout.subj_imgs, 
        subj_events = wf_firstlevel.get_subj_file.lzout.subj_events, 
    )
)

# add task - set_contrast
wf_firstlevel.add(
    set_contrast(
        name = "set_contrast",
        subj_id = wf_firstlevel.get_subj_file.lzout.subj_id,
        design_matrices = wf_firstlevel.get_firstlevel_dm.lzout.design_matrices
    )
)

# add task - firstlevel_estimation
wf_firstlevel.add(
    firstlevel_estimation(
        name = "firstlevel_estimation",
        subj_id = wf_firstlevel.get_subj_file.lzout.subj_id, 
        subj_imgs = wf_firstlevel.get_subj_file.lzout.subj_imgs, 
        subj_masks = wf_firstlevel.get_subj_file.lzout.subj_masks, 
        smoothing_fwhm = wf_firstlevel.lzin.smoothing_fwhm, 
        design_matrices = wf_firstlevel.get_firstlevel_dm.lzout.design_matrices, 
        contrasts = wf_firstlevel.set_contrast.lzout.contrasts
    )
)

# specify output
wf_firstlevel.set_output(
    [
        ('first_level_designmatrices', wf_firstlevel.get_firstlevel_dm.lzout.design_matrices),
        ('first_level_dm_paths', wf_firstlevel.get_firstlevel_dm.lzout.dm_paths),
        ('first_level_contrast', wf_firstlevel.set_contrast.lzout.contrasts),
        ('first_level_contrast_plot', wf_firstlevel.set_contrast.lzout.contrast_plot),
        ('first_level_model_list', wf_firstlevel.firstlevel_estimation.lzout.first_level_model),
        ('first_level_z_map_dict_list', wf_firstlevel.firstlevel_estimation.lzout.z_map_path_dict),
    ]
)
```

```{code-cell} ipython3
wf = Workflow(
    name='wf',
    input_spec=['subj_id','rawdata_url', 'fmriprep_url', 'smoothing_fwhm', 'output_dir'],
)

wf.split('subj_id', subj_id=[1,2,3])
wf.inputs.rawdata_url = 'https://github.com/OpenNeuroDerivatives/ds000001-fmriprep.git'
wf.inputs.fmriprep_url = 'https://github.com/OpenNeuroDatasets/ds000001.git'
wf.inputs.smoothing_fwhm = 5.0
wf.inputs.output_dir = workflow_out_dir

wf.add(
    get_data(
        name = "get_data",
        rawdata_url = wf.lzin.rawdata_url, 
        fmriprep_url = wf.lzin.fmriprep_url)
)

wf_firstlevel.inputs.subj_id = wf.lzin.subj_id,
wf_firstlevel.inputs.n_run = 3,
wf_firstlevel.inputs.tr = 2.3,
wf_firstlevel.inputs.n_scans = 300,
wf_firstlevel.inputs.hrf_model = 'glover',
wf_firstlevel.inputs.event_list = wf.get_data.lzout.event_list, 
wf_firstlevel.inputs.img_list = wf.get_data.lzout.img_list, 
wf_firstlevel.inputs.mask_list = wf.get_data.lzout.mask_list,
wf_firstlevel.inputs.smoothing_fwhm = wf.lzin.smoothing_fwhm,
wf_firstlevel.inputs.output_dir = wf.lzin.output_dir
wf.add(wf_firstlevel)

wf.combine('subj_id')

wf.set_output(
    [
        ('first_level_outputs', wf.wf_firstlevel.lzout.first_level_z_map_dict_list),
        
    ]
)
```

```{code-cell} ipython3
:tags: []

from pydra import Submitter

with Submitter(plugin='cf', n_procs=4) as submitter:
    submitter(wf)

results = wf.result()

print(results)
```

```{code-cell} ipython3

```