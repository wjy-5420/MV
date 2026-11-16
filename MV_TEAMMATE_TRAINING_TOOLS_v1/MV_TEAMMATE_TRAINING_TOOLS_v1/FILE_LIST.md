# 패키지 파일 목록

기존 Clone 기준 Commit: `8d01cc94fafc2a1cfef3237c7ca0ccb7c67a7d7f`

실제 모델 소스/가중치는 포함하지 않는다. SHA는 package.json 및 SHA256SUMS를 따른다.

|추가할 파일|바이트|SHA256|
|---|---:|---|
|`configs/training/efficientad_s.json`|821|`47a8d628812a33ed2bd9e7131c8bbfd8de07ec822046d0d6d11c5c8584092630`|
|`configs/training/hvq_trans.json`|790|`73146dd36ad7ade4c5d3be5448263e52a01118f857b52d2d4f34b6bf1be96a0c`|
|`configs/training/retinanet.json`|806|`17c111405703c9caa51d58260e047619b76b1ea62706c4de241e31e6b66345fe`|
|`configs/training/rtdetrv2.json`|847|`6b262a8c3ce2eb23172d770496a66e6c4f74e5b06ee6c971e103439c1cbb8988`|
|`configs/training/uniad.json`|789|`a72f052f7dcd73236db798b7cd430b25093fd992193500caf13f4b8c03b1cafb`|
|`configs/training/yolo11.json`|795|`d3904145c6b353c85662c856887cd2bfabeaf86ae9e9afcda04089c037c556d4`|
|`docs/experiments/TRAINING_IMPLEMENTATION_PROGRESS.md`|6486|`cf908d7fc1e5d2ab048fd297679f495ec802385abe2432f9734572fd4e820c32`|
|`docs/experiments/TRAINING_RUNNER_GUIDE.md`|9379|`3c48546a9e5e2ff71be7bbc3cb2c4a98821f4ad46ee75deea4d177a865c2d051`|
|`environments/README.md`|2459|`29b38bee58e72e8effc6c24b10b222b7abe90c6516e12323650643ae0c71e75e`|
|`mv_analysis/data/release_smoke_v2.py`|7411|`00e749cf5fb7baaa291fd3419213d2f66be5eb8023cdbf5d863c98720086da3d`|
|`mv_analysis/training/__init__.py`|132|`0dd2fecca0d88707362550b0de9d17b97dafdb9f37d8e74d74a2c78a3c9ca91c`|
|`mv_analysis/training/preflight.py`|7312|`02e318a2f71214e0194a6a9fa884e243ebc407a15c11a63aede4c647280442e7`|
|`mv_analysis/training/runner.py`|16049|`b0bcf0c3daf69c12f9c17c276913bde6abd62f0ea9efe234afb607ebc8b3e641`|
|`mv_analysis/training/settings.py`|8693|`ad438e25c809dd83d62017d836a842e608d312b041c4086f5bcb5dd16794658d`|
|`mv_analysis/training/state.py`|6421|`500e15c6d1761fbd990aa5cfafd7cd4e3d5368f1317a321f0c7f4cbbb9024d01`|
|`mv_analysis/training/worker.py`|4350|`75d34f316e0c03f8f8f9bfd9cc2c5d6a4b289f6b94b5452044aa0b083d3dd985`|
|`scripts/experiments/run_weekend.py`|317|`0777e7aea191e1487084b4d61afd5542c1c4f3a74ba5e7b1911bd8c7331ab45f`|
|`scripts/experiments/training_worker.py`|294|`3ad5fa940754c1d528f078c1f2072bfd01c1f13610b3cc9adae2a035795b025b`|
|`tests/test_team_data_release_v2.py`|8094|`dc2767f43396859eede473d5af4abc53b143e3d31e2844318c3f12d1dec3a90b`|
|`tests/test_training_runner.py`|14119|`f2daf15a789a2dcf148e925363cb11e597e1c6ed9846c4c260baa89ff9c6c28c`|

## 기존 Clone에서 재사용하는 파일

아래 파일은 ZIP에 복제하지 않고 설치 전에 해시를 검사한다.

- `configs/experiments/model_registry.json`
- `mv_analysis/__init__.py`
- `mv_analysis/adaptive_crop_gate.py`
- `mv_analysis/background_brightness.py`
- `mv_analysis/cli.py`
- `mv_analysis/core.py`
- `mv_analysis/crop_manifest.py`
- `mv_analysis/data/__init__.py`
- `mv_analysis/data/adapters.py`
- `mv_analysis/data/collate.py`
- `mv_analysis/data/datasets.py`
- `mv_analysis/data/manifest.py`
- `mv_analysis/data/release_builder.py`
- `mv_analysis/data/targets.py`
- `mv_analysis/data/team_release.py`
- `mv_analysis/data/team_release_v2.py`
- `mv_analysis/experiments/__init__.py`
- `mv_analysis/experiments/anomaly.py`
- `mv_analysis/experiments/benchmark.py`
- `mv_analysis/experiments/bridge.py`
- `mv_analysis/experiments/contracts.py`
- `mv_analysis/experiments/demo.py`
- `mv_analysis/experiments/detection.py`
- `mv_analysis/experiments/geometry.py`
- `mv_analysis/experiments/integration.py`
- `mv_analysis/experiments/pixels.py`
- `mv_analysis/experiments/recorder.py`
- `mv_analysis/experiments/reports.py`
- `mv_analysis/experiments/visualization.py`
- `mv_analysis/legacy_bridge.py`
- `mv_analysis/mask_validation.py`
- `mv_analysis/object_aware_crop.py`
- `mv_analysis/presence_heatmaps.py`
- `mv_analysis/regional_cli.py`
- `mv_analysis/report.py`
- `mv_analysis/schema.py`
- `mv_analysis/splits.py`
- `mv_analysis/stages.py`
- `mv_analysis/yolo_center_hybrid.py`
- `mv_analysis/yolo_crop_geometry.py`
- `pyproject.toml`
- `scripts/adaptive_crop_gate_experiment.py`
- `scripts/analyze_cross_relations.py`
- `scripts/analyze_preprocessing_evidence.py`
- `scripts/approve_mask_policy.py`
- `scripts/attach_resolution_metadata.py`
- `scripts/audit_content_hashes.py`
- `scripts/audit_full_crop_manifest.py`
- `scripts/audit_mask_integrity.py`
- `scripts/build_analysis_gallery.py`
- `scripts/build_preprocessing_review_package.py`
- `scripts/build_visual_crop_reports.py`
- `scripts/check_git_safety.py`
- `scripts/collect_model_preprocessing_sources.py`
- `scripts/compare_windows_brightness_groups.py`
- `scripts/compare_windows_checkpoints.py`
- `scripts/compare_windows_heatmaps.py`
- `scripts/compare_windows_results.py`
- `scripts/continue_image_analysis.py`
- `scripts/defect_relative_brightness.py`
- `scripts/enrich_analysis_results.py`
- `scripts/evaluate_yolo_center_hybrid.py`
- `scripts/experiments/compare_models.py`
- `scripts/experiments/generate_report.py`
- `scripts/experiments/smoke_test_reporting.py`
- `scripts/experiments/validate_real_input_contract.py`
- `scripts/export_detailed_pdf.py`
- `scripts/export_resolution_meeting_pdf.py`
- `scripts/extend_full_crop_report_manual.py`
- `scripts/finalize_adaptive_crop_report.py`
- `scripts/finalize_data_loader_validation.py`
- `scripts/finalize_preprocessing_review_package.py`
- `scripts/finish_image_analysis.py`
- `scripts/finish_spatial_analysis.py`
- `scripts/freeze_adaptive_crop_gate_recovery.py`
- `scripts/full_crop_gt_validation.py`
- `scripts/generate_full_crop_manifest.py`
- `scripts/inventory_dataset.py`
- `scripts/materialize_yolo_center_hybrid.py`
- `scripts/object_aware_crop_experiment.py`
- `scripts/object_crop_aligned_quality.py`
- `scripts/package_team_data_tools.py`
- `scripts/polish_adaptive_crop_figures.py`
- `scripts/prepare_manual_crop_review.py`
- `scripts/prepare_quality_manifest.py`
- `scripts/prepare_shared_report_links.py`
- `scripts/publish_analysis_branch.py`
- `scripts/record_preflight.py`
- `scripts/render_all_pdf_pages.py`
- `scripts/render_heatmaps.py`
- `scripts/render_pipeline_diagram.py`
- `scripts/render_regional_review.py`
- `scripts/report_adaptive_crop_gate.py`
- `scripts/report_background_brightness.py`
- `scripts/report_full_crop_manifest.py`
- `scripts/report_object_aware_crop.py`
- `scripts/report_resolution_detail.py`
- `scripts/report_yolo11_dynamic_crop.py`
- `scripts/resolution_batch.py`
- `scripts/resolve_filename_class_conflicts.py`
- `scripts/restore_windows_analysis_split.py`
- `scripts/resume_adaptive_crop_gate.py`
- `scripts/resume_preprocessing_review.py`
- `scripts/reuse_verified_product_boxes.py`
- `scripts/review_crop_gate_bboxes.py`
- `scripts/review_crop_ranges.py`
- `scripts/review_defect_resolution.py`
- `scripts/review_loader_findings.py`
- `scripts/review_product_bboxes.py`
- `scripts/run_presence_heatmaps.py`
- `scripts/select_final_preprocessing.py`
- `scripts/smoke_test_data_loaders.py`
- `scripts/smoke_test_shared_data.py`
- `scripts/team_data_release.py`
- `scripts/train_brightness_policies.py`
- `scripts/update_project_reports.py`
- `scripts/validate_artifacts.py`
- `scripts/validate_bbox_review_server.py`
- `scripts/validate_crop_range_review.py`
- `scripts/validate_detection_exports.py`
- `scripts/validate_environment.py`
- `scripts/validate_filename_class_resolution.py`
- `scripts/validate_final_preprocessing_decode.py`
- `scripts/validate_known_loader_cases.py`
- `scripts/validate_mask_thresholds.py`
- `scripts/validate_object_aware_crop.py`
- `scripts/validate_pdf_report.py`
- `scripts/validate_regional_results.py`
- `scripts/validate_resolution_results.py`
- `scripts/validate_yolo11_offline_experiment.py`
- `scripts/verify_bbox_backend_parity.py`
- `scripts/verify_source_preservation.py`
- `scripts/write_model_input_evidence.py`
- `scripts/yolo11_offline_crop_experiment.py`
- `tests/test_adaptive_crop_freeze_recovery.py`
- `tests/test_adaptive_crop_gate.py`
- `tests/test_background_brightness.py`
- `tests/test_crop_gate_review.py`
- `tests/test_crop_manifest.py`
- `tests/test_crop_range_review.py`
- `tests/test_data_loading.py`
- `tests/test_experiment_infrastructure.py`
- `tests/test_full_crop_generation.py`
- `tests/test_object_aware_crop.py`
- `tests/test_pipeline.py`
- `tests/test_preserved_contracts.py`
- `tests/test_product_bbox_review.py`
- `tests/test_product_bbox_review_http.py`
- `tests/test_shared_smoke.py`
- `tests/test_team_data_release.py`
- `tests/test_yolo_crop_geometry.py`
- `uv.lock`
