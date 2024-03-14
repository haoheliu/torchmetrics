# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List, Optional, Sequence, Union
import os
import random
import cv2
import traceback
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch import Tensor
from typing_extensions import Literal
from tqdm import tqdm
from torchmetrics import Metric
from torchmetrics.functional.multimodal.clip_score import _clip_score_update, _get_clip_model_and_processor, _get_image_feature, _get_text_feature
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_10
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CLIPScore.plot"]

if _SKIP_SLOW_DOCTEST and _TRANSFORMERS_GREATER_EQUAL_4_10:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor

    def _download_clip() -> None:
        _CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        _CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    if not _try_proceed_with_timeout(_download_clip):
        __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]
else:
    __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]

import pickle

def save_pickle(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(fname):
    with open(fname, "rb") as f:
        res = pickle.load(f)
    return res

class CLIPScore(Metric):
    r"""Calculates `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP Score is a reference free metric that can be used to evaluate the correlation between a generated caption for
    an image and the actual content of the image. It has been found to be highly correlated with human judgement. The
    metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual `CLIP`_ embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    .. note:: Metric is not scriptable

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``images`` (:class:`~torch.Tensor` or list of tensors): tensor with images feed to the feature extractor with. If
        a single tensor it should have shape ``(N, C, H, W)``. If a list of tensors, each tensor should have shape
        ``(C, H, W)``. ``C`` is the number of channels, ``H`` and ``W`` are the height and width of the image.
    - ``text`` (:class:`~str` or :class:`~list` of :class:`~str`): text to compare with the images, one for each image.

    As output of `forward` and `compute` the metric returns the following output

    - ``clip_score`` (:class:`~torch.Tensor`): float scalar tensor with mean CLIP score over samples

    Args:
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are:

            - `"openai/clip-vit-base-patch16"`
            - `"openai/clip-vit-base-patch32"`
            - `"openai/clip-vit-large-patch14-336"`
            - `"openai/clip-vit-large-patch14"`

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0

    Example:
        >>> import torch
        >>> from torchmetrics.multimodal.clip_score import CLIPScore
        >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
        >>> score = metric(torch.randint(255, (3, 224, 224), generator=torch.manual_seed(42)), "a photo of a cat")
        >>> score.detach()
        tensor(24.4255)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound = 100.0

    score: Tensor
    n_samples: Tensor

    def __init__(
        self,
        model_name_or_path: Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14",
        ] = "openai/clip-vit-large-patch14",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.processor = _get_clip_model_and_processor(model_name_or_path)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.txt_feature_cache = {}
        self.image_feature_cache = {}

        self.dataset_visual_feature_cache = {}

    def preprocess_image(self, image):
        preprocess = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
            ]
        )
        image = preprocess(image)
        image = (image * 255).to(torch.uint8)
        return image
    
    def extract_random_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame = random.randint(0, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        success, frame = cap.read()
        cap.release()
        if success:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            return None
        
    def build_dataset_cache(self, video_or_image_candidates_path_list):
        dataset_size = len(video_or_image_candidates_path_list)
        # If the dataset cache is already built, return
        if dataset_size in self.dataset_visual_feature_cache.keys():
            return
        self.dataset_visual_feature_cache[dataset_size] = {}
        self.dataset_visual_feature_cache[dataset_size]["feature"] = []
        self.dataset_visual_feature_cache[dataset_size]["path"] = []
        for candidate in tqdm(video_or_image_candidates_path_list):
            if candidate.endswith(".mp4"):
                frame_save_path = candidate+"_rand_frame.jpg"
                if os.path.exists(frame_save_path):
                    frame = Image.open(frame_save_path).convert("RGB")
                else:
                    try:
                        frame = self.extract_random_frame(candidate)
                    except Exception as e:
                        traceback.print_exc()
                        print("Error when extracting random frame", e, candidate)
                        continue
                    frame.save(frame_save_path)
            else:
                frame = Image.open(candidate).convert("RGB")
            cached_path = candidate+"_clip.pkl"
            if os.path.exists(cached_path):
                visual_feature = load_pickle(cached_path)
            else:
                visual_feature = _get_image_feature(self.preprocess_image(frame), self.model, self.processor)
                save_pickle(visual_feature, cached_path)
            self.dataset_visual_feature_cache[dataset_size]["feature"].append(visual_feature)
            self.dataset_visual_feature_cache[dataset_size]["path"].append(candidate)
        
        self.dataset_visual_feature_cache[dataset_size]["feature"] = torch.cat(self.dataset_visual_feature_cache[dataset_size]["feature"], dim=0)

    def get_device(self):
        return next(self.model.parameters()).device
    
    def query_resource_by_text(self, text, video_or_image_candidates_path_list, top_k=10, offset_rule = {}):
        self.build_dataset_cache(video_or_image_candidates_path_list)
        text_feature = _get_text_feature(text, self.model, self.processor)
        # Calculate the similarity between the text and each image
        candidate_feature = self.dataset_visual_feature_cache[len(video_or_image_candidates_path_list)]["feature"]
        candidate_path = self.dataset_visual_feature_cache[len(video_or_image_candidates_path_list)]["path"]
        score = 100*(candidate_feature * text_feature).sum(axis=-1)
        
        # This is to down-play some of the candidate based on quality
        offset = torch.zeros_like(score)
        for key in offset_rule:
            for i, path in enumerate(candidate_path):
                if key not in path:
                    continue
                offset[i] += offset_rule[key]
        score += offset

        ranks = torch.argsort(score, descending=True)
        result_filepath = []
        for i, index in enumerate(ranks):
            if i > top_k: break
            else:
                result_filepath.append(candidate_path[index])
        return result_filepath

    def update(self, images: Union[Tensor, List[Tensor]], text: Union[str, List[str]], image_src_path: str) -> None:
        """Update CLIP score on a batch of images and text.

        Args:
            images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors
            text: Either a single caption or a list of captions

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images and captions do not match

        """
        image_feature_path = image_src_path+"_clip.pkl"
        if image_feature_path in self.image_feature_cache.keys():
            img_features_prev = self.image_feature_cache[image_feature_path]
        else:
            if os.path.exists(image_feature_path):
                img_features_prev = load_pickle(image_feature_path)
            else:
                img_features_prev = None

        if text in self.txt_feature_cache.keys():
            txt_features_prev = self.txt_feature_cache[text]
        else:
            txt_features_prev = None

        score, n_samples, img_features, txt_features = _clip_score_update(images, text, img_features_prev ,txt_features_prev, self.model, self.processor)

        if txt_features_prev is None:
            self.txt_feature_cache[text] = txt_features
        if img_features_prev is None:
            save_pickle(img_features, image_feature_path)

        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self) -> Tensor:
        """Compute accumulated clip score."""
        return torch.max(self.score / self.n_samples, torch.zeros_like(self.score))

    def plot(self, val: Union[Tensor, Sequence[Tensor], None] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.multimodal.clip_score import CLIPScore
            >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            >>> metric.update(torch.randint(255, (3, 224, 224)), "a photo of a cat")
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.multimodal.clip_score import CLIPScore
            >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randint(255, (3, 224, 224)), "a photo of a cat"))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
