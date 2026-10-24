"""
# A python module that manipulates torch checkpoint file in a hacky way.
Each function should be used with caution and should be used only when thoughtfully considered.
---
Args:
    source_state_dict: the state_dict loaded using torch.load
    algo_state_dict: the algorithm state_dict summarized from algorithm as an example
---
Returns:
    new_state_dict: the state_dict that has been manipulated or directly saved as a checkpoint file.
"""

from collections import OrderedDict
from typing import Literal

import regex as re
import torch


def replace_encoder0(source_state_dict, algo_state_dict):
    print("\033[1;36m Replacing encoder.0 weights with untrained weights and avoid critic_encoder.0 \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if "critic_encoders.0" in key:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
        elif "encoders.0" in key:
            print(
                "key:", key, "shape:", algo_state_dict["model_state_dict"][key].shape, "using untrained module weights."
            )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def append_GRU_weights(source_state_dict, algo_state_dict):
    print("\033[1;36m Appending GRU weights to fit the new model \033[0m")
    print("\033[1;36m Operating on both actor and critic \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if ("memory_a" in key or "memory_c" in key) and "rnn" in key and "weight_ih" in key:
            print(
                "key:",
                key,
                "shape:",
                source_state_dict["model_state_dict"][key].shape,
                "is updated to shape:",
                algo_state_dict["model_state_dict"][key].shape,
            )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
            new_model_state_dict[key][:, : source_state_dict["model_state_dict"][key].shape[1]] = source_state_dict[
                "model_state_dict"
            ][key]
            new_model_state_dict[key][:, source_state_dict["model_state_dict"][key].shape[1] :] /= 10
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def append_GRU_weights_newStd(source_state_dict, algo_state_dict):
    return_ = append_GRU_weights(source_state_dict, algo_state_dict)
    print(
        "\033[1;36m Setting the std of the new actor to {} \033[0m".format(
            algo_state_dict["model_state_dict"]["std"].mean().cpu().item()
        )
    )
    return_["model_state_dict"]["std"][:] = algo_state_dict["model_state_dict"]["std"][:]
    return return_


def reinitialize_actor_critic_backbone(source_state_dict, algo_state_dict):
    print("\033[1;36m Reinitializing the actor and critic backbone \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if (
            "actor." in key
            or "critic." in key
            or "critics." in key
            or "memory_a" in key
            or "memory_c" in key
            or "std" in key
        ):
            if not key in source_state_dict["model_state_dict"]:
                print(
                    "key:",
                    key,
                    "shape:",
                    algo_state_dict["model_state_dict"][key].shape,
                    "using untrained module weights.",
                )
            else:
                print(
                    "key:",
                    key,
                    "shape:",
                    source_state_dict["model_state_dict"][key].shape,
                    "is updated to shape:",
                    algo_state_dict["model_state_dict"][key].shape,
                )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def ignore_missing_key(source_state_dict, algo_state_dict):
    """Ignore the missing critic mlp weights and use the initialized ones."""
    print("\033[1;36m Ignoring missing key and using the initialized weights \033[0m")
    new_model_state_dict = OrderedDict()
    missing_keys = []
    for key in algo_state_dict["model_state_dict"].keys():
        if key in source_state_dict["model_state_dict"]:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
            missing_keys.append(key)
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    print("\033[1;36m Missing keys: \033[0m", missing_keys)
    return new_state_dict


def fit_smaller_weight(
    source_state_dict: dict,
    algo_state_dict: dict,
    weight_name_regex: str = ".*",
    weight_match_mode: Literal["start", "end"] = "start",
):
    """To fix the weight matrix in algo_state_dict which is smaller than the one in source_state_dict,
    we will copy the part of the weight matrix from source_state_dict to algo_state_dict.
    ## Args:
        weight_name_regex: str
            The regex to match the weight name in algo_state_dict.
        weight_match_mode: Literal["start", "end"]
            If "start", weight_algo = weight_source[:weight_algo.shape[0], :weight_algo.shape[1]]
            If "end", weight_algo = weight_source[-weight_algo.shape[0]:, -weight_algo.shape[1]:]
    """
    print("\033[1;36m Fitting smaller weight matrix, matching \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if re.match(weight_name_regex, key):
            weight_algo = algo_state_dict["model_state_dict"][key]
            weight_source = source_state_dict["model_state_dict"][key]
            if weight_match_mode == "start":
                new_model_state_dict[key] = weight_source[: weight_algo.shape[0], : weight_algo.shape[1]]
            elif weight_match_mode == "end":
                new_model_state_dict[key] = weight_source[-weight_algo.shape[0] :, -weight_algo.shape[1] :]
            else:
                raise ValueError(f"Invalid weight_match_mode: {weight_match_mode}. Must be one of ['start', 'end'].")
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
    )
    for k in source_state_dict.keys():
        if k not in new_state_dict and not k.startswith("optimizer_state_dict"):
            new_state_dict[k] = source_state_dict[k]
    return new_state_dict


def newStd(
    source_state_dict: dict,
    algo_state_dict: dict,
):
    """Replicate everything except for policy std"""
    print(
        "\033[1;36m Setting the std of the new actor to {} \033[0m".format(
            algo_state_dict["model_state_dict"]["std"].mean().cpu().item()
        )
    )
    new_state_dict = OrderedDict()
    for state_dict_key in source_state_dict.keys():
        if state_dict_key == "model_state_dict":
            new_state_dict[state_dict_key] = OrderedDict()
            for model_state_dict_key in source_state_dict[state_dict_key].keys():
                if "std" == model_state_dict_key:
                    new_state_dict[state_dict_key][model_state_dict_key] = algo_state_dict["model_state_dict"][
                        model_state_dict_key
                    ]
                else:
                    new_state_dict[state_dict_key][model_state_dict_key] = source_state_dict[state_dict_key][
                        model_state_dict_key
                    ]
        else:
            new_state_dict[state_dict_key] = source_state_dict[state_dict_key]
    return new_state_dict


def supermimic_sidecar(
    source_state_dict: dict,
    algo_state_dict: dict,
    sidecar_path: str,
):
    """Convert a run of the former stand-alone SuperMimic trainer into the checkpoint of `SuperMimic`.

    The former trainer wrote the policy as an `OnPolicyRunner` checkpoint (`model_<it>.pt`, which is the source state
    dict) and the rest of the training state in a sidecar file (`supermimic_state_<it>.pt`). `SuperMimic` keeps all of
    it in one checkpoint (`SuperMimic.state_dict`), so this function moves the entries of the sidecar to their new
    keys. The replay storage was never saved.
    ## Args:
        sidecar_path: str
            Path of `supermimic_state_<it>.pt` of the same iteration as the source checkpoint.
    """
    print("\033[1;36m Converting the former SuperMimic checkpoint, training state from \033[0m", sidecar_path)
    sidecar = torch.load(sidecar_path, weights_only=True)
    assert sidecar["iteration"] == source_state_dict["iter"], "The sidecar is not of the iteration of the checkpoint."
    renamed_keys = {
        "wm": "world_model",
        "opt_wm": "world_model_optimizer",
        "critic": "critic",
        "opt_v": "critic_optimizer",
        "fail_head": "failure_head",
        "opt_fail": "failure_head_optimizer",
        "fail_head_start_it": "failure_head_start_iteration",
        "latent_dec": "latent_decoder",
    }
    new_state_dict = dict(source_state_dict)
    for sidecar_key, key in renamed_keys.items():
        if key in algo_state_dict:
            new_state_dict[key] = sidecar[sidecar_key]
        else:
            print("key:", sidecar_key, "is not used by the algorithm, ignored.")
    # the state of the task model has the same entries as the sidecar had at its top level
    new_state_dict["task_model"] = {key: sidecar[key] for key in algo_state_dict["task_model"]}
    return new_state_dict
