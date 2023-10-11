"""the original nougat do parallel on every page of a pdf
    we change the parallel on every pdf
"""
import sys
import re
from pathlib import Path
import logging
import argparse
from functools import partial
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device, default_batch_size
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible
import pypdf
import multiprocessing
from itertools import repeat
from typing import List, Optional, Union, Tuple, Callable
from PIL import Image
import cv2
import torch
from PIL import ImageOps
from torchvision.transforms.functional import resize, rotate
import numpy as np
from nougat.transforms import train_transform, test_transform
import json
from nougat.dataset.rasterize import rasterize_paper
from torch.utils.data import Dataset
from loguru import logger

logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=default_batch_size(),
        help="Batch size to use.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help="Path to checkpoint directory.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="0.1.0-small",
        help=f"Model tag to use.",
    )
    parser.add_argument("--out", "-o", type=Path, help="Output directory.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute already computed PDF, discarding previous predictions.",
    )
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Use float32 instead of bfloat16. Can speed up CPU conversion for some setups.",
    )
    parser.add_argument(
        "--no-markdown",
        dest="markdown",
        action="store_false",
        help="Do not add postprocessing step for markdown compatibility.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Add postprocessing step for markdown compatibility (default).",
    )
    parser.add_argument(
        "--no-skipping",
        dest="skipping",
        action="store_false",
        help="Don't apply failure detection heuristic.",
    )
    parser.add_argument(
        "--pages",
        "-p",
        type=str,
        help="Provide page numbers like '1-4,7' for pages 1 through 4 and page 7. Only works for single PDF input.",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        help="# of process for data loader.",
    )
    parser.add_argument(
        "--single-page",
        dest="single_page",
        action="store_true"
    )
    parser.add_argument(
        "--gpu-id",
        dest="gpu_id",
        type=int,
        default=0,
        help="the id of gpu that will be used"
    )
    parser.add_argument("pdf", nargs="+", type=Path, help="PDF(s) to process.")
    args = parser.parse_args()

    assert 0 <= args.gpu_id < torch.cuda.device_count()

    if args.checkpoint is None or not args.checkpoint.exists():
        args.checkpoint = get_checkpoint(args.checkpoint, model_tag=args.model)
    if args.out is None:
        logging.warning("No output directory. Output will be printed to console.")
    else:
        if not args.out.exists():
            logging.info("Output directory does not exist. Creating output directory.")
            args.out.mkdir(parents=True)
        if not args.out.is_dir():
            logging.error("Output has to be directory.")
            sys.exit(1)
    if len(args.pdf) == 1 and not args.pdf[0].suffix == ".pdf":
        # input is a list of pdfs
        try:
            pdfs_path = args.pdf[0]
            if pdfs_path.is_dir():
                args.pdf = list(pdfs_path.rglob("*.pdf"))
            else:
                args.pdf = [
                    Path(l) for l in open(pdfs_path).read().split("\n") if len(l) > 0
                ]
            logging.info(f"Found {len(args.pdf)} files.")
        except:
            pass
    if args.pages and len(args.pdf) == 1:
        pages = []
        for p in args.pages.split(","):
            if "-" in p:
                start, end = p.split("-")
                pages.extend(range(int(start) - 1, int(end)))
            else:
                pages.append(int(p) - 1)
        args.pages = pages
    else:
        args.pages = None
    return args

class Prepare():
    def __init__(
        self,
        input_size: List[int],
        align_long_axis: bool,
        window_size: int,
        encoder_layer: List[int],
        patch_size: int,
        embed_dim: int,
        num_heads: List[int],
    ):
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    @classmethod
    def from_config_file(cls, file: Path):
        config = json.loads(file.open().read())
        return cls(
            input_size = config["input_size"],
            align_long_axis = config["align_long_axis"],
            window_size = config["window_size"],
            encoder_layer = config["encoder_layer"],
            patch_size = config["patch_size"],
            embed_dim = config["embed_dim"],
            num_heads = config["num_heads"]
        )

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    def do(
        self, img: Image.Image, random_padding: bool = False
    ) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        if img is None:
            return
        # crop margins
        try:
            img = self.crop_margin(img.convert("RGB"))
        except OSError:
            # might throw an error for broken files
            return
        if img.height == 0 or img.width == 0:
            return
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.width > img.height)
            or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return test_transform(ImageOps.expand(img, padding))

class ImageDatasetWYQ():
    def __init__(self, img_list, prepare: Callable):
        self.img_list = img_list
        self.prepare = prepare

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_list[idx])
            return self.prepare(img)
        except Exception as e:
            logger.error(e)


class WYQDataset(Dataset):
    def __init__(self, pdf, dataset, is_single_page: bool=False):
        super().__init__()
        self.name = str(pdf)
        self.dataset = dataset
        if is_single_page:
            self.size = 1
        else:
            self.size = len(pypdf.PdfReader(pdf).pages)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i <= self.size and i >= 0:
            return self.dataset[i], self.name if i == self.size - 1 else ""
        else:
            raise IndexError


def _load_dataset(params):
    (out, recompute, checkpoint), pdf = params
    if not pdf.exists():
        return None, None
    if out:
        out_path = out / pdf.with_suffix(".mmd").name
        if out_path.exists() and not recompute:
            # logger.info(f"Skipping {pdf.name}, already computed. Run with --recompute to convert again.")
            return None, None
    try:
        prepare = Prepare.from_config_file(checkpoint.joinpath("config.json"))
        dataset = ImageDatasetWYQ(rasterize_paper(pdf, b_parallel=False), prepare.do)
        return dataset, pdf
    except pypdf.errors.PdfStreamError:
        logger.info(f"Could not load file {str(pdf)}.")
        return None, None
    except pypdf.errors.PyPdfError:  # catch all errors from PYPDF
        logger.info(f"Could not load file {str(pdf)}.")
        return None, None

def run(args, datasets):
    torch.cuda.empty_cache()

    dataloader = DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate
    )

    model = NougatModel.from_pretrained(args.checkpoint)
    model = move_to_device(model, bf16=not args.full_precision, cuda=args.batchsize > 0, gpu_id=args.gpu_id)
    model.eval()    
    
    predictions = []
    file_index = 0
    page_num = 0
    for i, (sample, is_last_page) in enumerate(tqdm(dataloader, desc=f"gpu {args.gpu_id}: png->mmd")):
        model_output = model.inference(
            image_tensors=sample, early_stopping=args.skipping
        )
        # check if model output is faulty
        for j, output in enumerate(model_output["predictions"]):
            # if page_num == 0:
            #     logger.info(
            #         "Processing file %s with %i pages"
            #         % (datasets[file_index].name, datasets[file_index].size)
            #     )
            page_num += 1
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
            elif args.skipping and model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    # If we end up here, it means the output is most likely not complete and was truncated.
                    # logger.warning(f"Skipping page {page_num} due to repetitions.")
                    predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                else:
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    predictions.append(
                        f"\n\n[MISSING_PAGE_EMPTY:{i*args.batchsize+j+1}]\n\n"
                    )
            else:
                if args.markdown:
                    output = markdown_compatible(output)
                predictions.append(output)
            if is_last_page[j]:
                out = "".join(predictions).strip()
                out = re.sub(r"\n{3,}", "\n\n", out).strip()
                if args.out:
                    out_path = args.out / Path(is_last_page[j]).with_suffix(".mmd").name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(out, encoding="utf-8")
                else:
                    print(out, "\n\n")
                predictions = []
                page_num = 0
                file_index += 1

def main():
    args = get_args()

    if args.batchsize <= 0:
        # set batch size to 1. Need to check if there are benefits for CPU conversion for >1
        args.batchsize = 1
    
    datasets = []
    with multiprocessing.Pool(args.n_proc) as pool:
        for dataset, pdf in tqdm(
            pool.imap(_load_dataset, zip(repeat((args.out, args.recompute, args.checkpoint)), args.pdf)), 
            total=len(args.pdf),
            desc=f"gpu {args.gpu_id}: pdf->png",
            mininterval=30
        ):
            if dataset is None: continue
            tmp = WYQDataset(pdf, dataset, args.single_page)
            datasets.append(tmp)
    
    if len(datasets) == 0:
        return

    while True:
        try:
            run(args, datasets)
            break
        except torch.cuda.OutOfMemoryError as e: # OOM error
            logger.error(e)
            assert args.batchsize > 1
            args.batchsize = args.batchsize // 2
            logger.info(f"rerun with batchsize {args.batchsize}")

    logger.info("Finish")    


if __name__ == "__main__":
    main()
