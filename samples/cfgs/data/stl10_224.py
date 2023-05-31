_base_ = [
    './pipelines/hflip_resize.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
        type='TVDatasetSplit',
        base='STL10',
        data_prefix='data/torchvision/stl10',
        split='train',
        pipeline=__train_pipeline,
        download=True,
    ),
    val=dict(
        type='TVDatasetSplit',
        base='STL10',
        split='test',
        data_prefix='data/torchvision/stl10',
        pipeline=__test_pipeline,
        download=True,
    ),
    test=dict(
        type='TVDatasetSplit',
        base='STL10',
        split='test',
        data_prefix='data/torchvision/stl10',
        pipeline=__test_pipeline,
        download=True,
    )
)
