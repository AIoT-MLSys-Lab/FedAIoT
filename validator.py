from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, TQDM_BAR_FORMAT
# from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.yolo.utils.torch_utils import de_parallel

from loaders.visdrone import NAMES


class YoloValidator:
    """
    BaseValidator
    A base class for creating validators.
    Attributes:
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        args (SimpleNamespace): Configuration for the validator.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        speed (float): Batch processing speed in seconds.
        jdict (dict): Dictionary to store validation results.
        save_dir (Path): Directory to save results.
    """

    def __init__(self, dataloader=None, nc=12, names=None, pbar=None, args=None, device='cuda'):
        """
        Initializes a BaseValidator instance.
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
        """
        if names is None:
            names = NAMES
        self.dataloader = dataloader
        self.pbar = pbar
        self.args = args or get_cfg(DEFAULT_CFG)
        self.model = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.device = device
        self.nc = 12
        self.names = names
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}
        # self.jdict = None
        self.metrics = DetMetrics(plot=False, names={k: v for k, v in enumerate(NAMES)})
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.device = 'cuda'

        # project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        # name = self.args.name or f'{self.args.mode}'
        # self.save_dir = save_dir or increment_path(Path(project) / name,
        #                                            exist_ok=self.args.exist_ok if RANK in (-1, 0) else True)
        # (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001

        # self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks

    # @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        # self.training = trainer is not None
        # if self.training:
        #     self.device = trainer.device
        #     self.data = trainer.data
        #     # model = trainer.model
        #     # self.args.half = self.device.type != 'cpu'  # force FP16 val during training
        #     # model = model.half() if self.args.half else model.float()
        #     self.model = model
        #     self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
        #     self.args.plots = trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
        #     model.eval()
        # else:
        # self.device = select_device(self.args.device, self.args.batch)
        # self.args.half &= self.device.type != 'cpu'
        # model = AutoBackend(model, device=self.device, dnn=self.args.dnn, data=self.args.data, fp16=self.args.half)
        self.model = model
        # stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        # imgsz = check_imgsz(self.args.imgsz, stride=stride)
        # if engine:
        #     self.args.batch = model.batch_size
        # else:
        # self.device = model.device
        # if not pt and not jit:
        #     self.args.batch = 1  # export.py models default to batch-size 1
        #     LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # if isinstance(self.args.data, str) and self.args.data.endswith('.yaml'):
        #     self.data = check_det_dataset(self.args.data)
        # elif self.args.task == 'classify':
        #     self.data = check_cls_dataset(self.args.data)
        # else:
        #     raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

        # if self.device.type == 'cpu':
        #     self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
        # if not pt:
        #     self.args.rect = False
        self.dataloader = self.dataloader  # or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

        model.eval()
        # model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        # dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
        # which may affect classification task since this arg is in yolov5/classify/val.py.
        # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        with torch.no_grad():
            for batch_i, batch in enumerate(bar):
                # self.run_callbacks('on_val_batch_start')
                self.batch_i = batch_i
                # preprocess
                # with dt[0]:
                batch = self.preprocess(batch)

                # inference
                # with dt[1]:
                preds = model(batch['img'])

                # loss
                # with dt[2]:
                #     if self.training:
                #         self.loss += trainer.criterion(preds, batch)[1]

                # postprocess
                # with dt[3]:
                preds = self.postprocess(preds)

                self.update_metrics(preds, batch)
                # if self.args.plots and batch_i < 3:
                #     self.plot_val_samples(batch, batch_i)
                #     self.plot_predictions(batch, preds, batch_i)

                # self.run_callbacks('on_val_batch_end')
        stats = self.get_stats()
        # self.check_stats(stats)
        self.print_results()
        # self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
        self.finalize_metrics()
        # self.run_callbacks('on_val_end')
        # if self.training:
        #     model.float()
        #     results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
        #     return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        # else:
        #     LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
        #                 tuple(self.speed.values()))
        #     if self.args.save_json and self.jdict:
        #         with open(str(self.save_dir / 'predictions.json'), 'w') as f:
        #             LOGGER.info(f'Saving {f.name}...')
        #             json.dump(self.jdict, f)  # flatten and save
        #         stats = self.eval_json(stats)  # update stats
        #     if self.args.plots or self.args.save_json:
        #         LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats

    # def run_callbacks(self, event: str):
    #     for callback in self.callbacks.get(event, []):
    #         callback(self)

    # def preprocess(self, batch):
    #     return batch

    def preprocess(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(self.device)

        nb = len(batch['img'])
        # self.lb = [torch.cat([batch['cls'], batch['bboxes']], dim=-1)[batch['batch_idx'] == i]
        #            for i in range(nb)]  # if self.args.save_hybrid else []  # for autolabelling
        self.lb = []
        return batch

    def init_metrics(self, model):
        # val = self.data.get(self.args.split, '')  # validation path
        # self.is_coco = isinstance(val, str) and val.endswith(f'coco{os.sep}val2017.txt')  # is COCO dataset
        self.class_map = list(range(12))
        # self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        # self.names = model.names
        # self.nc = 12
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)')

    def postprocess(self, preds):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=False,
                                        max_det=self.args.max_det)
        return preds

    def update_metrics(self, preds, batch):
        # Metrics
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            # if self.args.single_cls:
            #     pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch['ratio_pad'][si])  # native-space pred

            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            # # Save
            # if self.args.save_json:
            #     self.pred_to_json(predn, batch['im_file'][si])
            # if self.args.save_txt:
            #     file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
            #     self.save_one_txt(predn, self.args.save_conf, shape, file)

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            # print([type(stat) for stat in stats])
            self.metrics.names = {k: v for k, v in enumerate(self.names)}
            self.metrics.process(*stats)
        self.nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        # if self.args.plots:
        #     self.confusion_matrix.plot(save_dir=self.save_dir, names=list(self.names.values()))

    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)
# class DetectionValidator(BaseValidator):
#
#     def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None):
#         super().__init__(dataloader, save_dir, pbar, args)
#         self.args.task = 'detect'
#         self.is_coco = False
#         self.class_map = None
#         self.metrics = DetMetrics(save_dir=self.save_dir)
#         self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
#         self.niou = self.iouv.numel()


# def get_dataloader(self, dataset_path, batch_size):
#     # TODO: manage splits differently
#     # calculate stride - check if model is initialized
#     gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
#     return create_dataloader(path=dataset_path,
#                              imgsz=self.args.imgsz,
#                              batch_size=batch_size,
#                              stride=gs,
#                              hyp=vars(self.args),
#                              cache=False,
#                              pad=0.5,
#                              rect=self.args.rect,
#                              workers=self.args.workers,
#                              prefix=colorstr(f'{self.args.mode}: '),
#                              shuffle=False,
#                              seed=self.args.seed)[0] if self.args.v5loader else \
#         build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, names=self.data['names'],
#                          mode='val')[0]
#
# def plot_val_samples(self, batch, ni):
#     plot_images(batch['img'],
#                 batch['batch_idx'],
#                 batch['cls'].squeeze(-1),
#                 batch['bboxes'],
#                 paths=batch['im_file'],
#                 fname=self.save_dir / f'val_batch{ni}_labels.jpg',
#                 names=self.names)
#
# def plot_predictions(self, batch, preds, ni):
#     plot_images(batch['img'],
#                 *output_to_target(preds, max_det=15),
#                 paths=batch['im_file'],
#                 fname=self.save_dir / f'val_batch{ni}_pred.jpg',
#                 names=self.names)  # pred
#
# def save_one_txt(self, predn, save_conf, shape, file):
#     gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
#     for *xyxy, conf, cls in predn.tolist():
#         xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#         with open(file, 'a') as f:
#             f.write(('%g ' * len(line)).rstrip() % line + '\n')
#
# def pred_to_json(self, predn, filename):
#     stem = Path(filename).stem
#     image_id = int(stem) if stem.isnumeric() else stem
#     box = ops.xyxy2xywh(predn[:, :4])  # xywh
#     box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
#     for p, b in zip(predn.tolist(), box.tolist()):
#         self.jdict.append({
#             'image_id': image_id,
#             'category_id': self.class_map[int(p[5])],
#             'bbox': [round(x, 3) for x in b],
#             'score': round(p[4], 5)})
#
# def eval_json(self, stats):
#     if self.args.save_json and self.is_coco and len(self.jdict):
#         anno_json = self.data['path'] / 'annotations/instances_val2017.json'  # annotations
#         pred_json = self.save_dir / 'predictions.json'  # predictions
#         LOGGER.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
#         try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
#             check_requirements('pycocotools>=2.0.6')
#             from pycocotools.coco import COCO  # noqa
#             from pycocotools.cocoeval import COCOeval  # noqa
#
#             for x in anno_json, pred_json:
#                 assert x.is_file(), f'{x} file not found'
#             anno = COCO(str(anno_json))  # init annotations api
#             pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
#             eval = COCOeval(anno, pred, 'bbox')
#             if self.is_coco:
#                 eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
#             eval.evaluate()
#             eval.accumulate()
#             eval.summarize()
#             stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[:2]  # update mAP50-95 and mAP50
#         except Exception as e:
#             LOGGER.warning(f'pycocotools unable to run: {e}')
#     return
