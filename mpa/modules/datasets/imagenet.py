from mmcls.datasets.imagenet import ImageNet as OriginImageNet
from mmcls.datasets.builder import DATASETS


@DATASETS.register_module()
class ImageNet(OriginImageNet):

    def __init__(self, with_background=False, **kwargs):
        self.with_background = with_background
        super().__init__(**kwargs)

    def load_annotations(self):
        data_infos = super().load_annotations()
        if self.with_background:
            self.folder_to_idx = {key: value + 1 for key, value in self.folder_to_idx.items()}
            self.CLASSES.insert(0, "background")
            for data_info in data_infos:
                data_info['gt_label'] += 1
        return data_infos
