# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Serialize multi-model exports without reading or rewriting their weights."""

import json
import os

PIPELINE_FILENAME = "pipeline.onnx"


def _bind_state(model, config):
    inputs = {value.name for value in model.graph.input}
    outputs = {value.name for value in model.graph.output}
    existing = {binding.input_name: binding.output_name for binding in model.graph.persistent_bindings}
    for source, target in (
        ("past_key_names", "present_key_names"),
        ("past_value_names", "present_value_names"),
        ("past_conv_names", "present_conv_names"),
        ("past_recurrent_names", "present_recurrent_names"),
    ):
        past = config.get("inputs", {}).get(source)
        present = config.get("outputs", {}).get(target)
        if past is None or present is None:
            continue
        if isinstance(past, str):
            count = config.get("num_hidden_layers", 0)
            past = [past % i for i in range(count)] if "%d" in past else [past]
        if isinstance(present, str):
            count = config.get("num_hidden_layers", 0)
            present = [present % i for i in range(count)] if "%d" in present else [present]
        if len(past) != len(present):
            raise ValueError("Pipeline cache input/output declarations differ in length.")
        for input_name, output_name in zip(past, present):
            # Hybrid decoders may declare cache templates for inactive layers.
            if input_name not in inputs and output_name not in outputs:
                continue
            if input_name not in inputs or output_name not in outputs:
                raise ValueError(f"Missing pipeline state interface: {input_name!r} -> {output_name!r}.")
            if input_name in existing:
                if existing[input_name] != output_name:
                    raise ValueError(f"Conflicting state binding for {input_name!r}.")
                continue
            binding = model.graph.persistent_bindings.add()
            binding.input_name = input_name
            binding.output_name = output_name
            existing[input_name] = output_name


def save_pipeline(out_dir):
    """Add ``pipeline.onnx`` for representable multi-model GenAI exports.

    Component protos and the pipeline live in the same directory, so external
    locations (including child weight directories), offsets and lengths remain
    unchanged. Unsupported component roles are reported rather than omitted.
    """
    with open(os.path.join(out_dir, "genai_config.json")) as stream:
        config = json.load(stream)
    sections = {key: value for key, value in config["model"].items() if isinstance(value, dict) and "filename" in value}
    filenames = {os.path.normcase(os.path.abspath(os.path.join(out_dir, value["filename"]))) for value in sections.values()}
    path = os.path.join(out_dir, PIPELINE_FILENAME)
    canonical_path = os.path.normcase(os.path.abspath(path))
    if len(filenames) < 2:
        if canonical_path not in filenames and os.path.exists(path):
            os.remove(path)
        return None

    if canonical_path in filenames:
        raise ValueError(f"{PIPELINE_FILENAME!r} is reserved for the PipelineProto.")
    unsupported = sorted(set(sections) - {"embedding", "decoder", "prefill", "decode"})
    if unsupported:
        # Do not leave an earlier text-only bundle masquerading as this export.
        if os.path.exists(path):
            os.remove(path)
        print(
            "WARNING: No PipelineProto written: onnx-light PipelineProto needs "
            f"schema extensions for component ModelProto fields {unsupported!r} "
            "and their declared tensor/state interfaces. All standalone models "
            "and GenAI configuration are retained; no components were omitted "
            "from a purported complete pipeline."
        )
        return None

    sections = dict(sections)
    if "decoder" in sections:
        role = "decode" if "prefill" in sections else "prefill"
        if role in sections:
            raise ValueError(f"Ambiguous pipeline component role {role!r}.")
        sections[role] = sections.pop("decoder")
    if "prefill" not in sections and "decode" in sections:
        sections["prefill"] = sections.pop("decode")
    if "prefill" not in sections:
        raise ValueError("A PipelineProto requires a prefill/decoder component.")
    shared_decode = "decode" in sections and os.path.normpath(sections["decode"]["filename"]) == os.path.normpath(
        sections["prefill"]["filename"]
    )

    from onnx_light.onnx import PipelineProto, load_model
    from onnx_light.onnx_proto.verify import verify_pipeline

    pipeline = PipelineProto(name="mbext", producer_name="mbext", pipeline_version=1)
    for role, section in sections.items():
        if role == "decode" and shared_decode:
            continue
        filename = section["filename"]
        if os.path.dirname(filename) not in {"", "."}:
            raise ValueError(
                "Pipeline component models must be saved beside pipeline.onnx; " "external weights may use relative child directories."
            )
        model = load_model(os.path.join(out_dir, filename), load_external_data=False)
        _bind_state(model, section)
        getattr(pipeline, role).CopyFrom(model)
    if shared_decode:
        _bind_state(pipeline.prefill, sections["decode"])

    decoder = sections["prefill"]
    token_model = sections.get("embedding", decoder)
    pipeline.input_ids_name = token_model.get("inputs", {}).get("input_ids", "input_ids")
    generation_model = sections.get("decode", decoder)
    pipeline.logits_name = generation_model.get("outputs", {}).get("logits", "logits")
    for section in (decoder, generation_model):
        declared_logits = section.get("outputs", {}).get("logits")
        if declared_logits is not None and declared_logits != pipeline.logits_name:
            raise ValueError("PipelineProto requires identical prefill/decode logits names.")
    token_roles = ("embedding",) if pipeline.has_embedding() else ("prefill", "decode") if pipeline.has_decode() else ("prefill",)
    for role in token_roles:
        if pipeline.input_ids_name not in {value.name for value in getattr(pipeline, role).graph.input}:
            raise ValueError(f"Missing pipeline token input {pipeline.input_ids_name!r} in {role}.")
    logits_model = pipeline.decode if pipeline.has_decode() else pipeline.prefill
    if pipeline.logits_name not in {value.name for value in logits_model.graph.output}:
        raise ValueError(f"Missing pipeline logits output {pipeline.logits_name!r}.")
    for name in ("attention_mask", "position_ids"):
        names = {section["inputs"][name] for role, section in sections.items() if role != "embedding" and name in section.get("inputs", {})}
        if len(names) > 1:
            raise ValueError(f"PipelineProto requires identical prefill/decode {name} names.")
        if names:
            setattr(pipeline, f"{name}_name", names.pop())
    if pipeline.has_embedding():
        produced = {value.name for value in pipeline.embedding.graph.output}
        for role in ("prefill", "decode"):
            if role not in sections:
                continue
            component_role = "prefill" if role == "decode" and shared_decode else role
            consumed = {value.name for value in getattr(pipeline, component_role).graph.input}
            embedding_input = sections[role].get("inputs", {}).get("inputs_embeds")
            if embedding_input not in produced or embedding_input not in consumed:
                raise ValueError(f"Embedding outputs do not match {role} inputs; " "PipelineProto does not support interface renaming.")
    if pipeline.has_decode():
        prefill_outputs = {value.name for value in pipeline.prefill.graph.output}
        for binding in pipeline.decode.graph.persistent_bindings:
            if binding.output_name not in prefill_outputs:
                raise ValueError(f"Prefill does not produce decode state {binding.output_name!r}.")

    search = config.get("search", {})
    for name in ("max_new_tokens", "temperature", "top_k", "top_p"):
        if search.get(name) is not None:
            setattr(pipeline, name, search[name])
    if not search.get("do_sample", False):
        pipeline.temperature = 0.0
    if search.get("random_seed") is not None:
        pipeline.seed = search["random_seed"]
    for name in ("eos_token_id", "pad_token_id"):
        value = config["model"].get(name)
        if isinstance(value, list):
            if len(value) != 1:
                print(
                    f"WARNING: PipelineProto supports one {name}; using the first "
                    "ID. Multiple stop IDs require an onnx-light schema extension."
                )
            value = value[0] if value else None
        if value is not None:
            setattr(pipeline, name, value)

    verify_pipeline(pipeline)
    print(f"Saving PipelineProto in {path} (sharing standalone weight files)")
    pipeline.SerializeToFile(path)
    return path
