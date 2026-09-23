"""
PIE dataset preparation utilities for pedestrian crossing prediction.

This module prepares structured temporal features from the PIE dataset and
exposes them through a PyTorch ``Dataset`` for Transformer models.

The preprocessing pipeline is adapted from the public TrEP dataset pipeline:

    Zhang, Z., Tian, R. and Ding, Z. (2023).
    "TrEP: Transformer-Based Evidential Prediction for Pedestrian Intention
    with Uncertainty." AAAI Conference on Artificial Intelligence.

Original implementation:
    https://github.com/zzmonlyyou/TrEP/blob/main/dataset.py

The implementation has been modified for the experiments in this repository.
Key changes include:

- configuration-driven dataset and cache paths;
- support for cached raw PIE behaviour sequences and processed windows;
- relative bounding-box and pedestrian-centre representations;
- optional bounding-box area and aspect-ratio features;
- training-only horizontal mirroring for class augmentation;
- optional majority-class undersampling;
- configurable observation length, time-to-event and overlap; and
- compatibility with deterministic and evidential Transformer experiments.

The final dataset returns a temporal feature tensor of shape ``[T, F]``
and a binary crossing target for each observation window.
"""

from copy import deepcopy
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from PedestrianActionBenchmark.pie_data import PIE


class prepare_data:
    """Prepare and cache PIE sequences for pedestrian crossing prediction.

    Parameters
    dataset:
        Dataset name. Only PIE is supported by this implementation.
    configs:
        Resolved experiment configuration dictionary.
    config_path:
        Optional path to a YAML configuration file. Used when ``configs`` is
        not supplied.
    cache:
        Fallback value controlling whether generated data are cached.
    """

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        dataset: str = "PIE",
        configs: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
        cache: bool = True,
    ):
        self._dataset = dataset
        self._generator = None

        if dataset.upper() != "PIE":
            raise ValueError(
                f"Unsupported dataset: {dataset}. "
                "This implementation currently supports PIE only."
            )

        self.configs = self._load_config(configs=configs, config_path=config_path)

        data_config = self.configs.get("pie_data", {})
        model_opts = self.configs["model_opts"]
        data_opts = self.configs["data_opts"]

        self.cache_enabled = data_config.get("cache_enabled", cache)
        self.use_cached_sequences = data_config.get(
            "use_cached_sequences",
            True,
        )
        self.use_cached_processed = data_config.get(
            "use_cached_processed",
            True,
        )
        self.regenerate_sequences = data_config.get(
            "regenerate_sequences",
            False,
        )
        self.regenerate_processed = data_config.get(
            "regenerate_processed",
            False,
        )
        self.strict_cache = data_config.get("strict_cache", True)

        self.behaviour_cache_directory = self._resolve_directory(
            data_config["behaviour_cache_directory"],
        )
    
        self.processed_cache_directory = self._resolve_directory(
            data_config["processed_cache_directory"]
        )

        self.raw_data_directory = Path(
            data_config["raw_data_directory"]
        ).expanduser().resolve()

        model_opts["dataset"] = "pie"
        model_opts["generator"] = False

        maximum_tte = self._maximum_tte(model_opts["time_to_event"])
        data_opts["min_track_size"] = model_opts["obs_length"] + maximum_tte

        self._print_configuration(model_opts)

        self.beh_seq_train = self._load_or_create_behaviour_sequences(split="train")
        self.beh_seq_val = self._load_or_create_behaviour_sequences(split="val")
        self.beh_seq_test = self._load_or_create_behaviour_sequences(split="test")

        self.train_data = self.get_data(
            data_type="train",
            data_raw=self.beh_seq_train,
            model_opts=model_opts,
        )
        self.val_data = self.get_data(
            data_type="val",
            data_raw=self.beh_seq_val,
            model_opts=model_opts,
        )
        self.test_data = self.get_data(
            data_type="test",
            data_raw=self.beh_seq_test,
            model_opts=model_opts,
        )

    @staticmethod
    def _load_config(
        configs: dict[str, Any] | None,
        config_path: str | Path | None,
    ) -> dict[str, Any]:
        # Load and copy the experiment configuration.

        if configs is not None:
            if not isinstance(configs, dict):
                raise TypeError("`configs` must be a dictionary.")
            return deepcopy(configs)

        if config_path is not None:
            config_path = Path(config_path)
            with config_path.open("r", encoding="utf-8") as file:
                loaded = yaml.safe_load(file)

            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Invalid YAML configuration: {config_path}"
                )
            return loaded

        raise ValueError("Provide either `configs` or `config_path`.")

    @staticmethod
    def _resolve_directory(path: str | Path) -> Path:
        # Resolve and create a directory used by the data pipeline.

        directory = Path(path).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _maximum_tte(time_to_event: int | list[int] | tuple[int, ...]) -> int:
        # Return the largest configured time-to-event value.

        if isinstance(time_to_event, int):
            return time_to_event
        return max(time_to_event)

    def _print_configuration(self, model_opts: dict[str, Any]) -> None:
        # Print the data settings used for the current experiment.

        print("\nPIE data configuration")
        print("-" * 60)
        print(
            "Behaviour cache directory: "
            f"{self.behaviour_cache_directory}"
        )
        print(
            "Processed cache directory: "
            f"{self.processed_cache_directory}"
        )
        print(f"Raw PIE directory: {self.raw_data_directory}")
        print(f"Observation inputs: {model_opts['obs_input_type']}")
        print(f"Observation length: {model_opts['obs_length']}")
        print(f"Time to event: {model_opts['time_to_event']}")
        print(f"Overlap: {model_opts['overlap']}")
        print(
            "Normalize boxes: "
            f"{model_opts.get('normalize_boxes', False)}"
        )

    def _validate_split(self, split: str) -> str:
        # Validate a PIE split name

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid PIE split: {split}. "
                f"Expected one of {sorted(self.VALID_SPLITS)}."
            )
        return split

    def _behaviour_cache_path(self, split: str) -> Path:
        split = self._validate_split(split)
        return self.behaviour_cache_directory / f"beh_seq_{split}.pkl"

    def _processed_cache_path(self, split: str) -> Path:
        split = self._validate_split(split)
        return self.processed_cache_directory / f"PIE_{split}.pkl"

    @staticmethod
    def _load_pickle(path: Path):
        # Load a cached Python object
        with path.open("rb") as file:
            return pickle.load(file)

    @staticmethod
    def _save_pickle(data, path: Path) -> None:
        # Persist a Python object

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_or_create_behaviour_sequences(self, split: str):
        """Load cached PIE trajectories or generate them from raw PIE data."""

        split = self._validate_split(split)
        cache_path = self._behaviour_cache_path(split)

        can_use_cache = (
            self.use_cached_sequences
            and not self.regenerate_sequences
            and cache_path.exists()
        )

        if can_use_cache:
            print(
                f"Loading cached PIE {split} sequences: "
                f"{cache_path}"
            )
            return self._load_pickle(cache_path)

        if self.strict_cache:
            raise FileNotFoundError(
                "PIE sequence cache not found: "
                f"{cache_path}"
            )

        print(f"Generating PIE {split} behaviour sequences.")

        if not self.raw_data_directory.exists():
            raise FileNotFoundError(
                "Raw PIE directory not found: "
                f"{self.raw_data_directory}"
            )

        pie_dataset = PIE(data_path=str(self.raw_data_directory))
        sequence = pie_dataset.generate_data_trajectory_sequence(
            split,
            **self.configs["data_opts"],
        )

        if self.cache_enabled:
            self._save_pickle(sequence, cache_path)

        return sequence

    def get_data_sequence(
        self,
        data_type,
        data_raw,
        opts: dict[str, Any],
    ):
        """Generate fixed-length structured observation windows.

        Each raw pedestrian trajectory is converted into one or more temporal
        windows according to ``obs_length``, ``time_to_event`` and ``overlap``.

        When relative normalisation is enabled, bounding-box coordinates and
        pedestrian centres are expressed relative to the first raw observation.
        One initial observation is then removed from every feature sequence,
        leaving the final model input length one frame shorter than the
        configured raw observation length.

        Returns
        -------
        data:
            Dictionary containing aligned temporal features and metadata.
        neg_count:
            Number of non-crossing windows.
        pos_count:
            Number of crossing windows.
        """

        print("\n" + "#" * 37)
        print(f"Generating {data_type} observation windows")
        print("#" * 37)

        data = {
            "center": data_raw["center"].copy(),
            "box": data_raw["bbox"].copy(),
            "ped_id": data_raw["pid"].copy(),
            "crossing": data_raw["activities"].copy(),
        }

        if "obd_speed" not in data_raw:
            raise KeyError(
                "PIE behaviour sequence does not contain 'obd_speed'. "
                "Check the cached PIE sequence files."
            )

        data["speed"] = data_raw["obd_speed"].copy()

        balance = (
            opts.get("balance_data", False)
            if data_type == "train"
            else False
        )
        balance_strategy = (
            opts.get("balance_strategy", "hybrid")
            if balance
            else "none"
        )

        if balance_strategy != "none":
            image_width = data_raw["image_dimension"][0]
            self.balance_data_samples(
                data,
                img_width=image_width,
                strategy=balance_strategy,
            )

        # Preserve pixel-space boxes for derived size features.
        data["box_org"] = deepcopy(data["box"])
        data["tte"] = []

        obs_length = opts["obs_length"]
        time_to_event = opts["time_to_event"]
        normalize = opts.get("normalize_boxes", False)

        if isinstance(time_to_event, int):
            self._extract_fixed_tte_windows(
                data=data,
                obs_length=obs_length,
                time_to_event=time_to_event,
            )
        else:
            self._extract_tte_range_windows(
                data=data,
                obs_length=obs_length,
                time_to_event=time_to_event,
                overlap=opts["overlap"],
            )

        self._convert_to_arrays(
            data=data,
            normalize=normalize,
        )

        data["crossing"] = np.asarray(
            data["crossing"]
        )[:, 0, :]

        pos_count = int(np.count_nonzero(data["crossing"]))
        neg_count = int(len(data["crossing"]) - pos_count)

        print(
            f"Negative samples: {neg_count} | "
            f"Positive samples: {pos_count}"
        )

        return data, neg_count, pos_count

    @staticmethod
    def _extract_fixed_tte_windows(data, obs_length, time_to_event):
        # Extract one observation window per trajectory for a fixed TTE.

        for key in list(data.keys()):
            if key == "tte":
                continue

            for index in range(len(data[key])):
                data[key][index] = data[key][index][
                    -obs_length - time_to_event : -time_to_event
                ]

        number_of_samples = len(data["box"])
        data["tte"] = [[time_to_event]] * number_of_samples

    @staticmethod
    def _extract_tte_range_windows(data, obs_length, time_to_event, overlap):
        # Extract overlapping observation windows across a TTE interval.

        step = (
            obs_length
            if overlap == 0
            else int((1 - overlap) * obs_length)
        )
        step = max(step, 1)

        reference_boxes = data["box"]

        for key in list(data.keys()):
            if key == "tte":
                continue

            windows = []

            for sequence in data[key]:
                start_idx = (
                    len(sequence)
                    - obs_length
                    - time_to_event[1]
                )
                end_idx = (
                    len(sequence)
                    - obs_length
                    - time_to_event[0]
                )

                windows.extend(
                    sequence[index : index + obs_length]
                    for index in range(
                        start_idx,
                        end_idx + 1,
                        step,
                    )
                )

            data[key] = windows

        for sequence in reference_boxes:
            start_idx = (
                len(sequence)
                - obs_length
                - time_to_event[1]
            )
            end_idx = (
                len(sequence)
                - obs_length
                - time_to_event[0]
            )

            data["tte"].extend(
                [
                    [len(sequence) - (index + obs_length)]
                    for index in range(
                        start_idx,
                        end_idx + 1,
                        step,
                    )
                ]
            )

    @staticmethod
    def _convert_to_arrays(data, normalize):
        # Normalise temporal positions and convert sequences to arrays.

        if not normalize:
            for key in data:
                data[key] = np.asarray(data[key])
            return

        for key in data:
            if key == "tte":
                data[key] = np.asarray(data[key])
                continue

            if key in {"box", "center"}:
                normalised = [
                    np.subtract(
                        sequence[1:],
                        sequence[0],
                    ).tolist()
                    for sequence in data[key]
                ]
                data[key] = np.asarray(normalised)
            else:
                # Drop the initial raw observation so all features remain
                # aligned with the relative box/centre sequences.
                data[key] = np.asarray(
                    [
                        sequence[1:]
                        for sequence in data[key]
                    ]
                )

    def balance_data_samples(
        self,
        data,
        img_width,
        balance_tag="crossing",
        strategy="flip_only",
    ):
        '''Balance the training split.

        Parameters
        data:
            Aligned raw PIE trajectories.
        img_width:
            PIE frame width used to mirror x-coordinates.
        balance_tag:
            Label sequence used to determine the minority class.
        strategy:
            ``"flip_only"`` duplicates each minority-class trajectory using a
            horizontal geometric reflection. ``"hybrid"`` performs the same
            reflection and then randomly undersamples the remaining majority
            class until both classes contain the same number of trajectories.

        No image files are generated or modified by augmentation.'''
    

        valid_strategies = {"flip_only", "hybrid"}

        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown balance strategy: {strategy}. "
                f"Expected one of {sorted(valid_strategies)}."
            )

        labels = np.asarray(
            [target[0] for target in data[balance_tag]]
        )

        num_pos = int(np.count_nonzero(labels))
        num_neg = int(len(labels) - num_pos)

        print(
            f"Before augmentation: "
            f"Positive={num_pos}, Negative={num_neg}"
        )

        minority_label = 1 if num_neg > num_pos else 0
        original_num_samples = len(data[balance_tag])

        for index in range(original_num_samples):
            label = data[balance_tag][index][0][0]
            if label != minority_label:
                continue

            for key in data:
                if key == "center":
                    flipped = [
                        [img_width - center[0], center[1]]
                        for center in deepcopy(data[key][index])
                    ]
                    data[key].append(flipped)

                elif key == "box":
                    flipped = [
                        np.asarray(
                            [
                                img_width - box[2],
                                box[1],
                                img_width - box[0],
                                box[3],
                            ]
                        )
                        for box in deepcopy(data[key][index])
                    ]
                    data[key].append(flipped)

                else:
                    data[key].append(
                        deepcopy(data[key][index])
                    )

        labels = np.asarray(
            [target[0] for target in data[balance_tag]]
        )
        num_pos = int(np.count_nonzero(labels))
        num_neg = int(len(labels) - num_pos)

        print(
            f"After mirroring: "
            f"Positive={num_pos}, Negative={num_neg}"
        )

        if strategy == "flip_only":
            return

        majority_label = 0 if num_neg > num_pos else 1
        difference = abs(num_neg - num_pos)

        if difference == 0:
            return

        majority_indices = np.where(
            labels == majority_label
        )[0]

        rng = np.random.default_rng(42)
        remove_indices = set(
            rng.choice(
                majority_indices,
                size=difference,
                replace=False,
            ).tolist()
        )

        for key in data:
            sequence_data = data[key]
            data[key] = [
                sequence_data[index]
                for index in range(len(sequence_data))
                if index not in remove_indices
            ]

        final_labels = np.asarray(
            [target[0] for target in data[balance_tag]]
        )
        final_pos = int(np.count_nonzero(final_labels))
        final_neg = int(len(final_labels) - final_pos)

        print(
            f"After hybrid balancing: "
            f"Positive={final_pos}, Negative={final_neg}"
        )

    def get_data(
        self,
        data_type: str,
        data_raw: dict[str, Any],
        model_opts: dict[str, Any],
    ):
        # Construct processed data for one PIE split

        processed_cache_path = self._processed_cache_path(data_type)

        can_use_processed_cache = (
            self.use_cached_processed
            and not self.regenerate_processed
            and processed_cache_path.exists()
        )

        if can_use_processed_cache:
            print(
                f"Loading processed PIE {data_type} data: "
                f"{processed_cache_path}"
            )
            return self._load_pickle(processed_cache_path)

        print(
            f"Processing PIE {data_type} data from "
            "behaviour sequences."
        )

        data, neg_count, pos_count = self.get_data_sequence(
            data_type=data_type,
            data_raw=data_raw,
            opts=model_opts,
        )

        observation_data = []
        data_sizes = []
        data_types = []

        for input_type in model_opts["obs_input_type"]:
            if input_type not in data:
                raise KeyError(
                    f"Requested PIE input '{input_type}' was not "
                    f"generated. Available keys: {list(data.keys())}"
                )

            features = data[input_type]
            observation_data.append(features)
            data_sizes.append(features.shape[1:])
            data_types.append(input_type)

        processed_data = {
            "data": (
                observation_data,
                data["crossing"],
            ),
            "box_org": data["box_org"],
            "ped_id": data["ped_id"],
            "tte": data["tte"],
            "data_params": {
                "data_types": data_types,
                "data_sizes": data_sizes,
            },
            "count": {
                "neg_count": neg_count,
                "pos_count": pos_count,
            },
        }

        if self.cache_enabled:
            self._save_pickle(
                processed_data,
                processed_cache_path,
            )
            print(
                f"Saved processed PIE data: "
                f"{processed_cache_path}"
            )

        return processed_data


class PIECrossingDataset(Dataset):
    """PyTorch dataset for structured PIE crossing-prediction sequences.

    Parameters
    ----------
    set_data:
        Processed split produced by :class:`prepare_data`.
    feature_opts:
        Controls inclusion of centre, area and aspect-ratio features.
    expected_input_dim:
        Optional expected final feature dimension used as a consistency check.
    dtype:
        Tensor dtype returned for temporal feature sequences.

    Returns
    -------
    tuple
        ``(features, target)`` where ``features`` has shape ``[T, F]`` and
        ``target`` is a scalar binary class label.
    """

    def __init__(
        self,
        set_data,
        feature_opts=None,
        expected_input_dim=None,
        dtype=torch.float32,
    ):
        feature_opts = feature_opts or {}
        self.dtype = dtype

        features, labels = set_data["data"]
        self.targets = np.asarray(labels).reshape(-1).astype(np.int64)

        lookup = dict(
            zip(
                set_data["data_params"]["data_types"],
                features,
            )
        )

        self.features = self._build_tabular_features(
            lookup=lookup,
            box_org=set_data["box_org"],
            opts=feature_opts,
        )

        if (
            expected_input_dim is not None
            and self.features.shape[-1] != expected_input_dim
        ):
            raise ValueError(
                f"Expected {expected_input_dim} features, "
                f"got {self.features.shape[-1]}."
            )

    @staticmethod
    def _build_tabular_features(
        lookup, box_org, opts):
        # Construct the configured structured feature representation.

        required = {"box", "speed"}

        if opts.get("use_center", True):
            required.add("center")

        missing = sorted(required.difference(lookup))
        if missing:
            raise KeyError(
                f"Missing required processed features: {missing}"
            )

        selected = [
            np.asarray(
                lookup["box"],
                dtype=np.float32,
            ),
            np.asarray(
                lookup["speed"],
                dtype=np.float32,
            ),
        ]

        if opts.get("use_center", True):
            selected.append(
                np.asarray(
                    lookup["center"],
                    dtype=np.float32,
                )
            )

        pixel_boxes = np.asarray(
            box_org,
            dtype=np.float32,
        )

        width = np.maximum(
            pixel_boxes[..., 2] - pixel_boxes[..., 0],
            1e-6,
        )
        height = np.maximum(
            pixel_boxes[..., 3] - pixel_boxes[..., 1],
            1e-6,
        )

        if opts.get("use_area", False):
            area = np.log1p(
                width * height
            )[..., None]
            selected.append(area)

        if opts.get("use_aspect_ratio", False):
            aspect_ratio = (
                width / height
            )[..., None]
            selected.append(aspect_ratio)

        return np.concatenate(selected, axis=-1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        features = torch.as_tensor(self.features[index], dtype=self.dtype)
        target = torch.tensor(self.targets[index], dtype=torch.long)

        return features, target
