"""Gradio UI for the manual Qwen3-ASR workflow."""

from __future__ import annotations

import json

from phase1.qwen_asr.finetune import build_training_examples
from phase1.qwen_asr.workflow import (
    DEFAULT_QWEN_ASR_MODEL_ID,
    DEFAULT_QWEN_FORCED_ALIGNER_ID,
    download_qwen_assets,
    ensure_qwen_helper,
    format_command,
    qwen_default_asset_dir,
    qwen_helper_python,
    resolve_qwen_forced_aligner_path,
    resolve_qwen_model_path,
    run_qwen_pipeline,
    run_qwen_training_job,
)


def _format_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _format_command_result(result, *, label: str) -> str:
    lines = [f"{label}: {format_command(result.command)}"]
    output = result.combined_output()
    if output:
        lines.append("")
        lines.append(output)
    if result.returncode != 0:
        raise RuntimeError("\n".join(lines))
    return "\n".join(lines)


def build_qwen_asr_ui():
    import gradio as gr

    def bootstrap(
        model_id: str,
        model_dir: str,
        install_helper: bool,
        download_model: bool,
        download_aligner: bool,
        forced_aligner_id: str,
        forced_aligner_dir: str,
    ) -> tuple[str, str]:
        messages: list[str] = []
        if install_helper:
            result = ensure_qwen_helper()
            messages.append(_format_command_result(result, label="helper install"))
        payload: dict[str, object] = {
            "helper_python": str(qwen_helper_python()),
        }
        if download_model:
            payload.update(
                download_qwen_assets(
                    model_id=model_id or DEFAULT_QWEN_ASR_MODEL_ID,
                    model_dir=model_dir or None,
                    forced_aligner_id=forced_aligner_id or DEFAULT_QWEN_FORCED_ALIGNER_ID,
                    forced_aligner_dir=forced_aligner_dir or None,
                    download_forced_aligner=download_aligner,
                )
            )
        messages.append(_format_json(payload))
        return "\n\n".join(messages), str(payload.get("model_path") or resolve_qwen_model_path({}))

    def build_dataset(
        audio_path: str,
        transcript_path: str,
        output_dir: str,
        language: str,
        prompt: str,
        min_clip_seconds: float,
        max_clip_seconds: float,
        max_gap_seconds: float,
        min_keep_seconds: float,
        eval_ratio: float,
        duration_limit: float | None,
        model_path: str,
    ) -> tuple[str, str, str]:
        manifest = build_training_examples(
            audio_path=audio_path,
            transcript_source=transcript_path,
            output_dir=output_dir,
            language=language,
            prompt=prompt,
            min_clip_seconds=min_clip_seconds,
            max_clip_seconds=max_clip_seconds,
            max_gap_seconds=max_gap_seconds,
            min_keep_seconds=min_keep_seconds,
            eval_ratio=eval_ratio,
            duration_limit=duration_limit,
            model_path=model_path or resolve_qwen_model_path({}),
        )
        return (
            _format_json(manifest),
            str(manifest["artifacts"]["train_jsonl"]),
            str(manifest["artifacts"].get("eval_jsonl") or ""),
        )

    def train(
        train_file: str,
        eval_file: str,
        output_dir: str,
        model_path: str,
        batch_size: int,
        grad_acc: int,
        learning_rate: float,
        epochs: float,
        device: str,
    ) -> str:
        result = run_qwen_training_job(
            train_file=train_file,
            eval_file=eval_file or None,
            output_dir=output_dir,
            model_path=model_path or resolve_qwen_model_path({}),
            batch_size=batch_size,
            grad_acc=grad_acc,
            learning_rate=learning_rate,
            epochs=epochs,
            device=device or None,
        )
        return _format_command_result(result, label="training")

    def run_pipeline_ui(
        audio_path: str,
        output_dir: str,
        language: str,
        model_path: str,
        context: str,
        use_forced_aligner: bool,
        forced_aligner_path: str,
        min_speakers: float | None,
        max_speakers: float | None,
        duration_limit: float | None,
        skip_diarization: bool,
    ) -> str:
        result = run_qwen_pipeline(
            audio_path=audio_path,
            output_dir=output_dir,
            language=language,
            model_path=model_path or resolve_qwen_model_path({}),
            context=context,
            forced_aligner_model_path=forced_aligner_path or resolve_qwen_forced_aligner_path({}),
            use_forced_aligner=use_forced_aligner,
            min_speakers=None if min_speakers in (None, 0) else int(min_speakers),
            max_speakers=None if max_speakers in (None, 0) else int(max_speakers),
            duration_limit=duration_limit,
            skip_diarization=skip_diarization,
        )
        return _format_json(result)

    with gr.Blocks(title="Qwen3-ASR Workbench") as demo:
        gr.Markdown(
            "## Qwen3-ASR Workbench\n"
            "Manual bootstrap, dataset prep, supervised fine-tuning, and phase1 pipeline runs."
        )
        with gr.Tab("Bootstrap"):
            with gr.Row():
                model_id = gr.Textbox(label="Model ID", value=DEFAULT_QWEN_ASR_MODEL_ID)
                model_dir = gr.Textbox(label="Local Model Dir", value=str(qwen_default_asset_dir(DEFAULT_QWEN_ASR_MODEL_ID)))
            with gr.Row():
                forced_aligner_id = gr.Textbox(label="Forced Aligner ID", value=DEFAULT_QWEN_FORCED_ALIGNER_ID)
                forced_aligner_dir = gr.Textbox(
                    label="Local Aligner Dir",
                    value=str(qwen_default_asset_dir(DEFAULT_QWEN_FORCED_ALIGNER_ID)),
                )
            with gr.Row():
                install_helper = gr.Checkbox(label="Install Helper Env", value=True)
                download_model = gr.Checkbox(label="Download Model", value=True)
                download_aligner = gr.Checkbox(label="Download Forced Aligner", value=True)
            bootstrap_btn = gr.Button("Bootstrap", variant="primary")
            bootstrap_log = gr.Textbox(label="Bootstrap Log", lines=18)
            model_path_state = gr.Textbox(label="Resolved Model Path", value=resolve_qwen_model_path({}))
            bootstrap_btn.click(
                bootstrap,
                inputs=[
                    model_id,
                    model_dir,
                    install_helper,
                    download_model,
                    download_aligner,
                    forced_aligner_id,
                    forced_aligner_dir,
                ],
                outputs=[bootstrap_log, model_path_state],
            )

        with gr.Tab("Dataset"):
            dataset_audio = gr.Textbox(label="Audio Path")
            dataset_transcript = gr.Textbox(label="Gemini Transcript Path")
            dataset_output = gr.Textbox(label="Dataset Output Dir")
            with gr.Row():
                dataset_language = gr.Textbox(label="Language", value="ru")
                dataset_prompt = gr.Textbox(label="Prompt", value="")
            with gr.Row():
                min_clip = gr.Number(label="Min Clip Seconds", value=6.0)
                max_clip = gr.Number(label="Max Clip Seconds", value=24.0)
                max_gap = gr.Number(label="Max Gap Seconds", value=0.75)
                min_keep = gr.Number(label="Min Keep Seconds", value=1.0)
            with gr.Row():
                eval_ratio = gr.Number(label="Eval Ratio", value=0.1)
                duration_limit = gr.Number(label="Duration Limit", value=None)
                dataset_model_path = gr.Textbox(label="Launcher Model Path", value=resolve_qwen_model_path({}))
            dataset_btn = gr.Button("Build Dataset", variant="primary")
            dataset_manifest = gr.Textbox(label="Dataset Manifest", lines=18)
            train_file = gr.Textbox(label="Train JSONL")
            eval_file = gr.Textbox(label="Eval JSONL")
            dataset_btn.click(
                build_dataset,
                inputs=[
                    dataset_audio,
                    dataset_transcript,
                    dataset_output,
                    dataset_language,
                    dataset_prompt,
                    min_clip,
                    max_clip,
                    max_gap,
                    min_keep,
                    eval_ratio,
                    duration_limit,
                    dataset_model_path,
                ],
                outputs=[dataset_manifest, train_file, eval_file],
            )

        with gr.Tab("Train"):
            train_model_path = gr.Textbox(label="Model Path", value=resolve_qwen_model_path({}))
            train_output = gr.Textbox(label="Output Dir")
            with gr.Row():
                batch_size = gr.Number(label="Batch Size", value=8)
                grad_acc = gr.Number(label="Grad Acc", value=8)
                learning_rate = gr.Number(label="Learning Rate", value=2e-5)
                epochs = gr.Number(label="Epochs", value=1.0)
            train_device = gr.Dropdown(choices=["", "cpu", "cuda", "mps"], value="", label="Device Override")
            train_btn = gr.Button("Start Training", variant="primary")
            train_log = gr.Textbox(label="Training Log", lines=18)
            train_btn.click(
                train,
                inputs=[
                    train_file,
                    eval_file,
                    train_output,
                    train_model_path,
                    batch_size,
                    grad_acc,
                    learning_rate,
                    epochs,
                    train_device,
                ],
                outputs=train_log,
            )

        with gr.Tab("Run Pipeline"):
            pipeline_audio = gr.Textbox(label="Audio Path")
            pipeline_output = gr.Textbox(label="Output Dir")
            with gr.Row():
                pipeline_language = gr.Dropdown(choices=["ru", "uk", "en"], value="ru", label="Language")
                pipeline_model_path = gr.Textbox(label="Model Path", value=resolve_qwen_model_path({}))
            pipeline_context = gr.Textbox(label="Context Prompt", value="", lines=3)
            with gr.Row():
                use_forced_aligner = gr.Checkbox(label="Use Forced Aligner", value=bool(resolve_qwen_forced_aligner_path({})))
                forced_aligner_path = gr.Textbox(label="Forced Aligner Path", value=resolve_qwen_forced_aligner_path({}) or "")
            with gr.Row():
                min_speakers = gr.Number(label="Min Speakers", value=0)
                max_speakers = gr.Number(label="Max Speakers", value=0)
                pipeline_duration_limit = gr.Number(label="Duration Limit", value=None)
            skip_diarization = gr.Checkbox(label="Skip Diarization", value=False)
            pipeline_btn = gr.Button("Run Pipeline", variant="primary")
            pipeline_log = gr.Textbox(label="Pipeline Result", lines=18)
            pipeline_btn.click(
                run_pipeline_ui,
                inputs=[
                    pipeline_audio,
                    pipeline_output,
                    pipeline_language,
                    pipeline_model_path,
                    pipeline_context,
                    use_forced_aligner,
                    forced_aligner_path,
                    min_speakers,
                    max_speakers,
                    pipeline_duration_limit,
                    skip_diarization,
                ],
                outputs=pipeline_log,
            )

    return demo


def launch_qwen_asr_ui(*, server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False) -> None:
    """Launch the local Gradio workbench."""

    demo = build_qwen_asr_ui()
    demo.launch(server_name=server_name, server_port=server_port, share=share)
