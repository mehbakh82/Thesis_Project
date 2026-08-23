"""thesis-s2s command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="thesis-s2s")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bake = sub.add_parser("bakeoff")
    p_bake.add_argument("--no-hf", action="store_true")
    sub.add_parser("cosyvoice-probe")
    p_s3 = sub.add_parser("s3-inventory")
    p_s3.add_argument("--out", type=Path, default=Path("results/s3_inventory.json"))
    p_prep = sub.add_parser("prepare-youtube")
    p_prep.add_argument("--csv-dir", type=Path, required=True)
    p_prep.add_argument("--wav-dir", type=Path, default=None)
    p_prep.add_argument("--remote-chunks", type=str, default=None)
    p_prep.add_argument("--out-dir", type=Path, required=True)
    p_prep.add_argument("--manifest-dir", type=Path, required=True)
    p_prep.add_argument("--min-episode-rows", type=int, default=50)
    p_ing = sub.add_parser("ingest-youtube")
    p_ing.add_argument("--max-hours", type=float, default=10.0)
    p_ing.add_argument("--max-episodes", type=int, default=None)
    p_ing.add_argument("--no-resume", action="store_true")
    p_asr = sub.add_parser("batch-reasr")
    p_asr.add_argument("--in-jsonl", type=Path, required=True)
    p_asr.add_argument("--out-jsonl", type=Path, required=True)
    p_asr.add_argument("--limit", type=int, default=None)
    p_asr.add_argument("--no-resume", action="store_true")
    p_asr.add_argument("--prefer-http-nemo", action=argparse.BooleanOptionalAction, default=False)
    p_asr.add_argument("--workers", type=int, default=8)
    p_filt = sub.add_parser("filter-corpus")
    p_filt.add_argument("--in-jsonl", type=Path, required=True)
    p_filt.add_argument("--out-jsonl", type=Path, required=True)
    p_filt.add_argument("--min-hours", type=float, default=100)
    p_filt.add_argument("--max-hours", type=float, default=200)
    p_filt.add_argument("--require-teacher", action="store_true")
    p_audit = sub.add_parser("audit-corpus")
    p_audit.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifests/filtered.jsonl")
    )
    p_audit.add_argument("--out", type=Path, default=Path("results/corpus_audit.json"))
    p_audit.add_argument("--no-check-files", action="store_true")
    p_alignment = sub.add_parser("audit-alignment")
    p_alignment.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifests/filtered_nemo_teacher.jsonl"),
    )
    p_alignment.add_argument(
        "--out", type=Path, default=Path("results/caption_alignment_audit.json")
    )
    p_conv = sub.add_parser("build-conversations")
    p_conv.add_argument("--in-jsonl", type=Path, required=True)
    p_conv.add_argument(
        "--out-jsonl", type=Path, default=Path("data/processed/manifests/conversations.jsonl")
    )
    p_conv.add_argument("--clips-dir", type=Path, default=Path("data/processed/conversations"))
    p_conv.add_argument("--max-hours", type=float, default=200.0)
    p_conv_audit = sub.add_parser("audit-conversations")
    p_conv_audit.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifests/conversations.jsonl")
    )
    p_conv_audit.add_argument("--out", type=Path, default=Path("results/conversation_audit.json"))
    p_conv_audit.add_argument("--no-check-files", action="store_true")
    p_omni_export = sub.add_parser("export-omni2-data")
    p_omni_export.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifests/conversations.jsonl")
    )
    p_omni_export.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/manifests/llama_omni2_questions.json"),
    )
    p_conv_plan = sub.add_parser("plan-conversation-corpus")
    p_conv_plan.add_argument("--target-hours", type=float, default=180.0)
    p_conv_plan.add_argument("--min-hours", type=float, default=100.0)
    p_conv_plan.add_argument("--max-hours", type=float, default=200.0)
    p_conv_plan.add_argument("--max-channel-share", type=float, default=0.55)
    p_conv_plan.add_argument("--reserve-hours", type=float, default=120.0)
    p_conv_plan.add_argument("--reserve-max-hours", type=float, default=150.0)
    p_ep = sub.add_parser("prepare-conversation-episodes")
    p_ep.add_argument(
        "--selection",
        type=Path,
        default=Path("data/processed/manifests/youtube_conversation_selection.jsonl"),
    )
    p_ep.add_argument("--out-root", type=Path, default=Path("data/processed/conversation_windows"))
    p_ep.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("data/processed/manifests/conversation_episode_windows.jsonl"),
    )
    p_ep.add_argument("--max-episodes", type=int, default=None)
    p_ep.add_argument("--max-source-hours", type=float, default=None)
    p_ep.add_argument("--window-seconds", type=float, default=900.0)
    p_ep.add_argument("--min-chunk-coverage", type=float, default=0.95)
    p_ep.add_argument("--no-resume", action="store_true")
    p_ep_prepared_audit = sub.add_parser("audit-prepared-episodes")
    p_ep_prepared_audit.add_argument(
        "--selection",
        type=Path,
        default=p_ep.get_default("selection"),
    )
    p_ep_prepared_audit.add_argument(
        "--manifest",
        type=Path,
        default=p_ep.get_default("out_jsonl"),
    )
    p_ep_prepared_audit.add_argument(
        "--out",
        type=Path,
        default=Path("results/prepared_episode_audit.json"),
    )
    p_ep_prepared_audit.add_argument("--min-hours", type=float, default=100.0)
    p_ep_prepared_audit.add_argument("--max-hours", type=float, default=240.0)
    p_ep_prepared_audit.add_argument("--no-check-files", action="store_true")
    p_ep_diar = sub.add_parser("diarize-conversation-episodes")
    p_ep_diar.add_argument(
        "--in-jsonl",
        type=Path,
        default=Path("data/processed/manifests/conversation_episode_windows.jsonl"),
    )
    p_ep_diar.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("data/processed/manifests/conversation_episode_windows_diarized.jsonl"),
    )
    p_ep_diar.add_argument("--limit", type=int, default=None)
    p_ep_diar.add_argument("--no-resume", action="store_true")
    p_ep_merge = sub.add_parser("merge-conversation-windows")
    p_ep_merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    p_ep_merge.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path(
            "data/processed/manifests/conversation_episode_windows_diarized_combined.jsonl"
        ),
    )
    p_ep_audit = sub.add_parser("audit-diarized-episodes")
    p_ep_audit.add_argument("--manifest", type=Path, default=p_ep_diar.get_default("out_jsonl"))
    p_ep_audit.add_argument("--out", type=Path, default=Path("results/diarized_episode_audit.json"))
    p_ep_audit.add_argument("--min-hours", type=float, default=100.0)
    p_ep_audit.add_argument("--max-hours", type=float, default=200.0)
    p_qa_sample = sub.add_parser("sample-conversation-qa")
    p_qa_sample.add_argument(
        "--in-jsonl",
        type=Path,
        default=p_ep_diar.get_default("out_jsonl"),
    )
    p_qa_sample.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/manifests/conversation_manual_qa.csv"),
    )
    p_qa_sample.add_argument("--per-stratum", type=int, default=5)
    p_qa_apply = sub.add_parser("apply-conversation-qa")
    p_qa_apply.add_argument(
        "--in-jsonl",
        type=Path,
        default=p_ep_diar.get_default("out_jsonl"),
    )
    p_qa_apply.add_argument(
        "--qa-csv",
        type=Path,
        default=p_qa_sample.get_default("out"),
    )
    p_qa_apply.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("data/processed/manifests/conversation_episode_windows_reviewed.jsonl"),
    )
    p_qa_apply.add_argument("--report", type=Path, default=Path("results/manual_qa_report.json"))
    p_rights_create = sub.add_parser("create-conversation-rights-review")
    p_rights_create.add_argument(
        "--in-jsonl",
        type=Path,
        default=p_qa_apply.get_default("out_jsonl"),
    )
    p_rights_create.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/manifests/conversation_rights_review.csv"),
    )
    p_rights_create.add_argument("--overwrite", action="store_true")
    p_rights_apply = sub.add_parser("apply-conversation-rights-review")
    p_rights_apply.add_argument(
        "--in-jsonl",
        type=Path,
        default=p_qa_apply.get_default("out_jsonl"),
    )
    p_rights_apply.add_argument(
        "--rights-csv",
        type=Path,
        default=p_rights_create.get_default("out"),
    )
    p_rights_apply.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("data/processed/manifests/conversation_episode_windows_approved.jsonl"),
    )
    p_rights_apply.add_argument(
        "--report", type=Path, default=Path("results/conversation_rights_report.json")
    )
    p_scale = sub.add_parser("scale-corpus")
    p_scale.add_argument("--min-hours", type=float, default=100)
    p_scale.add_argument("--max-hours", type=float, default=200)
    p_scale.add_argument("--reasr-limit", type=int, default=None)
    p_syn = sub.add_parser("synth-duplex")
    p_syn.add_argument("--hours", type=float, default=2.0)
    p_fac = sub.add_parser("data-factory")
    p_fac.add_argument("--max-hours", type=float, default=10.0)
    p_fac.add_argument("--max-episodes", type=int, default=None)
    p_fac.add_argument("--reasr-limit", type=int, default=None)
    p_fac.add_argument("--synthetic-hours", type=float, default=2.0)
    p_barge = sub.add_parser("train-bargein")
    p_barge.add_argument("--recorded-jsonl", type=Path, default=None)
    p_barge.add_argument("--n-per-class", type=int, default=36)
    sub.add_parser("train-s2s-smoke")
    p_tr = sub.add_parser("train-s2s")
    p_tr.add_argument("--steps", type=int, default=None)
    p_tr.add_argument("--epochs", type=float, default=None)
    p_tr.add_argument("--max-steps", type=int, default=None)
    p_tr.add_argument("--jsonl", type=Path, default=None)
    p_tr.add_argument("--no-llm", action="store_true")
    p_tr.add_argument("--allow-experimental", action="store_true")
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--path", choices=["A", "B", "both"], default="both")
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--record", action="store_true")
    p_serve.add_argument("--study", action="store_true")
    p_serve.add_argument(
        "--retention",
        choices=["audio", "features", "metrics"],
        default=None,
        help="study persistence; defaults to audio with --record, otherwise features with --study",
    )
    p_serve.add_argument("--session-id", default="S001")
    p_serve.add_argument("--speaker-id", default="P01")
    p_serve.add_argument("--age-bin", default="under_60", choices=["under_60", "60plus"])
    p_serve.add_argument("--detector", default="gbdt", choices=["gbdt", "energy"])
    sub.add_parser("export-recordings")
    p_study_summary = sub.add_parser("study-summary")
    p_study_summary.add_argument("--out", type=Path, default=Path("results/eval/human_study.json"))
    p_gpu = sub.add_parser("gpu-preflight")
    p_gpu.add_argument("--out", type=Path, default=Path("results/hardware/gpu_preflight.json"))
    p_upstream = sub.add_parser("verify-upstreams")
    p_upstream.add_argument("--checkouts-root", type=Path, default=None)
    p_snapshot = sub.add_parser("release-snapshot")
    p_snapshot.add_argument("--out", type=Path, default=Path("results/release/snapshot.json"))
    sub.add_parser("refresh-dataset-card")
    args = parser.parse_args(argv)

    if args.cmd == "bakeoff":
        from thesis_s2s.bakeoff.run import run_bakeoff

        print(json.dumps(run_bakeoff(allow_hf=not args.no_hf), indent=2, default=str)[:5000])
    elif args.cmd == "cosyvoice-probe":
        from thesis_s2s.bakeoff.codecs import probe_cosyvoice2
        from thesis_s2s.config import project_root
        from thesis_s2s.metrics import write_json

        payload = probe_cosyvoice2()
        write_json(project_root() / "results" / "bakeoff" / "cosyvoice_probe.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif args.cmd == "s3-inventory":
        from thesis_s2s.data.s3_inventory import inventory

        print(json.dumps(inventory(args.out), indent=2, default=str)[:4000])
    elif args.cmd == "prepare-youtube":
        from thesis_s2s.data.prepare_youtube import prepare

        print(
            json.dumps(
                prepare(
                    args.csv_dir,
                    args.wav_dir,
                    args.out_dir,
                    args.manifest_dir,
                    remote_chunks=args.remote_chunks,
                    min_episode_rows=args.min_episode_rows,
                ),
                indent=2,
            )
        )
    elif args.cmd == "ingest-youtube":
        from thesis_s2s.data.ingest import run_ingest

        print(
            json.dumps(
                run_ingest(
                    max_hours=args.max_hours,
                    max_episodes=args.max_episodes,
                    resume=not args.no_resume,
                ),
                indent=2,
                default=str,
            )[:5000]
        )
    elif args.cmd == "batch-reasr":
        from thesis_s2s.data.batch_reasr import run_manifest

        print(
            json.dumps(
                run_manifest(
                    args.in_jsonl,
                    args.out_jsonl,
                    args.limit,
                    resume=not args.no_resume,
                    prefer_http_nemo=args.prefer_http_nemo,
                    workers=args.workers,
                ),
                indent=2,
            )
        )
    elif args.cmd == "filter-corpus":
        from thesis_s2s.data.filter_corpus import filter_hours

        print(
            json.dumps(
                filter_hours(
                    args.in_jsonl,
                    args.out_jsonl,
                    args.min_hours,
                    args.max_hours,
                    require_teacher=args.require_teacher,
                ),
                indent=2,
            )
        )
    elif args.cmd == "audit-corpus":
        from thesis_s2s.data.audit import audit_manifest

        report = audit_manifest(args.manifest, args.out, check_files=not args.no_check_files)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "audit-alignment":
        from thesis_s2s.data.audit import audit_caption_alignment

        report = audit_caption_alignment(args.manifest, args.out)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "build-conversations":
        from thesis_s2s.data.conversation import build_conversation_manifest

        report = build_conversation_manifest(
            args.in_jsonl,
            args.out_jsonl,
            args.clips_dir,
            max_hours=args.max_hours,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "audit-conversations":
        from thesis_s2s.data.conversation import audit_conversation_manifest

        report = audit_conversation_manifest(
            args.manifest,
            args.out,
            check_files=not args.no_check_files,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "export-omni2-data":
        from thesis_s2s.data.conversation import export_llama_omni2_questions

        print(
            json.dumps(
                export_llama_omni2_questions(args.manifest, args.out), indent=2, ensure_ascii=False
            )
        )
    elif args.cmd == "plan-conversation-corpus":
        from thesis_s2s.data.conversation_selection import run_conversation_selection

        report = run_conversation_selection(
            target_hours=args.target_hours,
            min_hours=args.min_hours,
            max_hours=args.max_hours,
            max_channel_share=args.max_channel_share,
            reserve_hours=args.reserve_hours,
            reserve_max_hours=args.reserve_max_hours,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "prepare-conversation-episodes":
        from thesis_s2s.data.episode_prepare import prepare_selected_episodes

        report = prepare_selected_episodes(
            args.selection,
            args.out_root,
            args.out_jsonl,
            max_episodes=args.max_episodes,
            max_source_hours=args.max_source_hours,
            window_seconds=args.window_seconds,
            min_chunk_coverage=args.min_chunk_coverage,
            resume=not args.no_resume,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "audit-prepared-episodes":
        from thesis_s2s.data.episode_prepare import audit_prepared_episode_windows

        report = audit_prepared_episode_windows(
            args.selection,
            args.manifest,
            args.out,
            min_hours=args.min_hours,
            max_hours=args.max_hours,
            check_files=not args.no_check_files,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "diarize-conversation-episodes":
        from thesis_s2s.data.diarize import diarize_episode_manifest

        report = diarize_episode_manifest(
            args.in_jsonl,
            args.out_jsonl,
            limit=args.limit,
            resume=not args.no_resume,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "merge-conversation-windows":
        from thesis_s2s.data.diarize import merge_conversation_window_manifests

        report = merge_conversation_window_manifests(args.inputs, args.out_jsonl)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "audit-diarized-episodes":
        from thesis_s2s.data.diarize import audit_diarized_windows

        report = audit_diarized_windows(
            args.manifest,
            args.out,
            min_hours=args.min_hours,
            max_hours=args.max_hours,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "sample-conversation-qa":
        from thesis_s2s.data.manual_qa import sample_manual_qa

        report = sample_manual_qa(
            args.in_jsonl,
            args.out,
            per_stratum=args.per_stratum,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "apply-conversation-qa":
        from thesis_s2s.data.manual_qa import apply_manual_qa

        report = apply_manual_qa(
            args.in_jsonl,
            args.qa_csv,
            args.out_jsonl,
            report_path=args.report,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "create-conversation-rights-review":
        from thesis_s2s.data.rights import create_conversation_rights_review

        report = create_conversation_rights_review(
            args.in_jsonl,
            args.out,
            overwrite=args.overwrite,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "apply-conversation-rights-review":
        from thesis_s2s.data.rights import apply_conversation_rights_review

        report = apply_conversation_rights_review(
            args.in_jsonl,
            args.rights_csv,
            args.out_jsonl,
            report_path=args.report,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "scale-corpus":
        from thesis_s2s.data.factory import scale_corpus

        print(
            json.dumps(
                scale_corpus(
                    min_hours=args.min_hours, max_hours=args.max_hours, reasr_limit=args.reasr_limit
                ),
                indent=2,
                default=str,
            )[:8000]
        )
    elif args.cmd == "synth-duplex":
        from thesis_s2s.config import project_root
        from thesis_s2s.data.filter_corpus import write_synthetic_duplex

        root = project_root()
        print(
            json.dumps(
                write_synthetic_duplex(
                    root / "data" / "processed" / "synthetic",
                    root / "data" / "processed" / "manifests" / "synthetic_duplex.jsonl",
                    target_hours=args.hours,
                ),
                indent=2,
            )
        )
    elif args.cmd == "data-factory":
        from thesis_s2s.data.factory import run_factory

        print(
            json.dumps(
                run_factory(
                    max_audio_hours=args.max_hours,
                    max_episodes=args.max_episodes,
                    reasr_limit=args.reasr_limit,
                    synthetic_hours=args.synthetic_hours,
                ),
                indent=2,
                default=str,
            )[:6000]
        )
    elif args.cmd == "train-bargein":
        from thesis_s2s.bargein.train import train_and_eval

        print(
            json.dumps(
                train_and_eval(n_per_class=args.n_per_class, recorded_jsonl=args.recorded_jsonl),
                indent=2,
            )
        )
    elif args.cmd == "train-s2s-smoke":
        from thesis_s2s.model.llama_omni2 import train_smoke

        print(json.dumps(train_smoke(), indent=2))
    elif args.cmd == "train-s2s":
        from thesis_s2s.model.llama_omni2 import train_s2s

        print(
            json.dumps(
                train_s2s(
                    steps=args.steps,
                    epochs=args.epochs,
                    max_steps=args.max_steps,
                    jsonl=args.jsonl,
                    load_llm=not args.no_llm,
                    allow_experimental=args.allow_experimental,
                ),
                indent=2,
                default=str,
            )
        )
    elif args.cmd == "eval":
        from thesis_s2s.eval.bench import (
            run_interrupt_bench,
            run_latency_bench,
            run_reply_wer,
            write_eval_summary,
        )

        paths = ["A", "B"] if args.path == "both" else [args.path]
        latency = None
        for p in paths:
            latency = run_latency_bench(path=p)
        assert latency is not None
        interrupt = run_interrupt_bench()
        wer_report = run_reply_wer()
        write_eval_summary(latency, interrupt, wer_report)
        print(
            json.dumps(
                {"latency": latency, "interrupt": interrupt, "wer": wer_report},
                indent=2,
                default=str,
            )[:6000]
        )
    elif args.cmd == "serve":
        import uvicorn

        from thesis_s2s.runtime.duplex import build_app

        app = build_app(
            record=args.record,
            study=args.study,
            retention=args.retention,
            session_id=args.session_id,
            speaker_id=args.speaker_id,
            age_bin=args.age_bin,
            detector_name=args.detector,
        )
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.cmd == "export-recordings":
        from thesis_s2s.runtime.session_log import SessionStore

        print(json.dumps(SessionStore().export_manifest(), indent=2, ensure_ascii=False))
    elif args.cmd == "study-summary":
        from thesis_s2s.runtime.session_log import SessionStore

        print(json.dumps(SessionStore().study_summary(args.out), indent=2, ensure_ascii=False))
    elif args.cmd == "gpu-preflight":
        from thesis_s2s.repro import gpu_preflight

        print(json.dumps(gpu_preflight(args.out), indent=2, ensure_ascii=False))
    elif args.cmd == "verify-upstreams":
        from thesis_s2s.repro import verify_upstream_lock

        print(
            json.dumps(
                verify_upstream_lock(checkouts_root=args.checkouts_root),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.cmd == "release-snapshot":
        from thesis_s2s.repro import release_snapshot

        print(json.dumps(release_snapshot(args.out), indent=2, ensure_ascii=False))
    elif args.cmd == "refresh-dataset-card":
        from thesis_s2s.data.factory import refresh_dataset_card

        path = refresh_dataset_card()
        print(str(path))


if __name__ == "__main__":
    main()
