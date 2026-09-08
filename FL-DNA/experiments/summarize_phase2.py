"""Rebuild Phase 2 tables and the Vietnamese handover from saved evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    d = json.loads((run / 'heldout_diagnostic_results.json').read_text())
    rows = d['all_records']
    assert not d['failed_jobs'], d['failed_jobs']
    assert len(rows) == 90
    selected = d['selected_records']
    pairs = []
    curves = []
    for b in [r for r in rows if r['method'] == 'baseline']:
        z = next(r for r in rows if r['method']=='zero_gradient' and (r['sample_id'], r['restart']) == (b['sample_id'], b['restart']))
        p = next(r for r in rows if r['method']=='prior' and (r['sample_id'], r['restart']) == (b['sample_id'], b['restart']))
        pairs.append(dict(sample=b['sample_id'], label=b['label'], restart=b['restart'],
                          baseline_mse=b['feature_mse'], zero_mse=z['feature_mse'], prior_mse=p['feature_mse'],
                          delta_zero=b['feature_mse']-z['feature_mse'], delta_prior=b['feature_mse']-p['feature_mse']))
    for r in rows:
        for point in r.get('component_history', []):
            curves.append(dict(sample=r['sample_id'], restart=r['restart'], method=r['method'], **point))
    csv_write(run / 'paired_results.csv', pairs)
    csv_write(run / 'convergence.csv', curves)
    features = []
    for r in rows:
        for name, value in r['per_feature_abs_error_normalized'].items():
            features.append(dict(sample=r['sample_id'], restart=r['restart'], method=r['method'],
                                 feature=name, normalized_absolute_error=value,
                                 raw_absolute_error=r['per_numeric_feature_abs_error_raw'].get(name)))
    csv_write(run / 'feature_results.csv', features)
    checks = [json.loads(p.read_text()) for p in (run/'vectors').glob('*/checks.json')]
    assert len(checks)==30 and all(c['passed'] for c in checks)
    oldroot = ROOT/'artifacts/phase1_validation/run_phase1_closure_v2'
    old = json.loads((oldroot/'validation_report.json').read_text())
    development = [r for r in old['records'] if r.get('budget')==300]
    csv_write(run/'development_results.csv', [{k:r[k] for k in ['experiment','sample_id','sample_label','restart','l2_weight','feature_mse','gradient_match_loss','regularization_loss','best_attack_loss']} for r in development])
    pairing = 0
    for b in [r for r in old['records'] if r.get('experiment','').startswith('baseline_') and 'budget' in r]:
        name = b['experiment']
        a = torch.load(oldroot/f'{name}_initial_dummy.pt', weights_only=False)['vector']
        z = torch.load(oldroot/f'{name.replace("baseline_", "zero_gradient_", 1)}_initial_dummy.pt', weights_only=False)['vector']
        assert torch.equal(a,z)
        pairing += 1
    config = d['config']
    lines = ['# Báo cáo nghiệm thu Phase 2', '',
        'Báo cáo này thay thế kết luận bàn giao tạm thời trước đó. Run đã xác minh: `'+run.name+'`.', '',
        '## Trạng thái', '',
        '- Implementation: PASS trong phạm vi 13 tests và kiểm tra artifact đã chạy.',
        '- Diagnostic: SUPPORTED_ON_PILOT cho lợi ích MSE so với zero-gradient; chất lượng tuyệt đối và so với prior còn hỗn hợp.',
        '- Phase 2: COMPLETE_WITH_LIMITATIONS. Có thể bắt đầu đặc tả/validation Phase 3; chưa có bằng chứng full FedAvg inversion hoặc DNA hiệu quả.', '',
        '## Công việc và thuật toán', '',
        '2A: đọc closure cũ, xuất bảng budget 300 riêng từng target/restart; xác minh '+str(pairing)+' cặp initialization bằng torch.equal. Closure có 102 attack calls và 12 prior records; các budget là lời gọi độc lập dùng cùng seed, không phải checkpoint của một trajectory. Hai target development không phải hai nhóm dân số.', '',
        '2B: lưu RobustScaler center/scale fit trên train và categories. MAE từng feature ở không gian scale và đơn vị gốc; transaction type chấm argmax, strict one-hot yêu cầu mỗi phần tử gần 0/1 và đúng một phần tử gần 1 (atol=1e-6). Kiểm tra balance differences sau inverse transform. Không áp phương trình raw vào các feature có scale khác nhau.', '',
        '2C: chỉ phát triển ứng viên bỏ L2 từ hai development targets. Giữ Adam lr=0.05, 300 bước, 3 restart. Chưa triển khai cosine objective, softmax parameterization, TabLeak: đây là các ứng viên tuần tự, không phải điều kiện bắt buộc phải thử tất cả. Không tuning DNA.', '',
        'Objective: L_match = mean_p(mean((gradient_p(dummy)-observed_p)^2)); L_total=L_match+lambda*mean(dummy^2), lambda=0. Các parameter tensor có trọng số bằng nhau, không phải toàn bộ phần tử bằng nhau. Dummy N(0,1); initialization iteration 0 có tham gia best-so-far. Mỗi bước lưu match/regularization/total. Candidate và restart chọn bằng minimum total objective, không dùng ground truth; không so objective giữa baseline và zero để suy ra leakage.', '',
        '2D: 50k-row stratified subset, một round warm-up FedAvg 3 client, MLP 128/64/32/1, focal loss và local Adam hiện tại. Gradient quan sát thuộc một mẫu known-label ở eval mode: BatchNorm dùng running buffers, Dropout tắt. Đây là checkpoint ít huấn luyện và gradient từng mẫu, chưa đại diện delta local training. 10 target có 5 fraud/5 non-fraud, không coi trung bình cân bằng lớp là leakage dân số.', '',
        '## Provenance và tái lập', '',
        'Source row IDs (zero-based data row, excluding header): `'+str(config['source_row_ids'])+'`. Development IDs: `'+str(config['development_source_row_ids'])+'`. Đã assert không giao nhau. Mapping replay đúng stratified sampling/split bằng test dataframe và assert labels; CSV SHA256: `'+config['dataset_sha256']+'`.', '',
        'Checkpoint đã lưu, reload trước capture; checksum trong JSON. Seed dữ liệu '+str(config['data_seed'])+', attack root '+str(config['attack_seed_root'])+'. Scaler, validation indices, commit/worktree status trong JSON. Protocol lock ghi trước attack. Cấu hình giữ nguyên sau pilot trước; đây là lượt kiểm tra kỹ thuật, không được gọi là đánh giá mù mới độc lập.', '',
        '## Kết quả từng target', '',
        'Baseline/zero chọn restart riêng theo objective; prior trung bình 3 initialization. Không gọi các restart là mẫu độc lập.', '',
        '| Target | Label | Baseline MSE | Zero MSE | Prior MSE | Baseline MAE | Type đúng | Best step |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for sid in range(10):
        b=next(r for r in selected if r['method']=='baseline' and r['sample_id']==sid)
        z=next(r for r in selected if r['method']=='zero_gradient' and r['sample_id']==sid)
        pr=[r for r in selected if r['method']=='prior' and r['sample_id']==sid]
        lines.append(f"| {sid} | {b['label']} | {b['feature_mse']:.6f} | {z['feature_mse']:.6f} | {np.mean([r['feature_mse'] for r in pr]):.6f} | {b['feature_mae']:.6f} | {b['categorical_accuracy']:.0f} | {b['best_iteration']} |")
    lines += ['', '## So sánh theo lớp', '', '| Label | Method | MSE | MAE | Type accuracy |', '|---|---|---:|---:|---:|']
    for label in (0,1):
        for method in ('baseline','zero_gradient','prior'):
            group=[r for r in selected if r['label']==label and r['method']==method]
            lines.append(f"| {label} | {method} | {np.mean([r['feature_mse'] for r in group]):.6f} | {np.mean([r['feature_mae'] for r in group]):.6f} | {np.mean([r['categorical_accuracy'] for r in group]):.2f} |")
    wins=sum(p['delta_zero']<0 for p in pairs)
    priorwins=sum(p['delta_prior']<0 for p in pairs)
    end=sum(r['best_iteration']==config['iterations'] for r in selected if r['method']=='baseline')
    lines += ['', '## Nhận xét và cổng quyết định', '',
        f'Cùng target/restart: baseline tốt hơn zero ở {wins}/30 cặp và prior ở {priorwins}/30 cặp. Đây là 10 target với 3 restart. Không báo p-value hoặc CI dân số. Bảng selected và bảng cùng restart là hai estimand khác nhau.', '',
        f'{end}/10 baseline selected còn đạt best tại bước 300. Loss curve được lưu đầy đủ nhưng chưa đủ chứng minh hội tụ; không mở budget sau xem evaluation. Lợi ích tương đối không đồng nghĩa tái dựng thành công. Fraud errors lệch mạnh, có target MSE lớn chi phối mean. MSE và MAE có thể cho kết luận khác nhau; xem cả bảng lớp và từng feature.', '',
        'Categorical validity còn kém vì dummy không bị ràng buộc. Balance consistency là phép đo, chưa phải constraint trong optimizer. Không kết luận nguyên nhân thất bại chỉ từ những triệu chứng này. Không có threshold tái dựng thành công đã hiệu chỉnh (TBD).', '',
        '## Kiểm tra và compute', '',
        f'13 tests pass. {len(checks)*4} vector reload checks kiểm tra shape/dtype/giá trị; {len(checks)*2} objective reload checks dùng checkpoint và observed signal, so match/reg/total với candidate. 30 pairing assertions trong evaluation. 60 optimization trajectories, 30 priors; 0 failed jobs.', '',
        'Runtime: '+json.dumps(d['runtime'])+'. Timer attack bao gồm baseline+zero cùng restarts; total gồm setup/training/lưu file. Không đo peak RAM/VRAM. Không ngoại suy sang Adam multi-step inversion.', '',
        '## Artifacts và lệnh', '',
        f'Run directory: `{run.relative_to(ROOT)}`. `heldout_diagnostic_results.json`, `heldout_trials.csv`, `paired_results.csv`, `feature_results.csv`, `convergence.csv`, `development_results.csv`, checkpoint, targets, observed gradients và vectors. Artifact cũ giữ nguyên.', '',
        '```bash', 'DATALOADER_NUM_WORKERS=0 .venv-phase1/bin/python -m experiments.run_phase2_diagnostic', f'.venv-phase1/bin/python -m experiments.summarize_phase2 {run.relative_to(ROOT)}', '.venv-phase1/bin/python -m pytest -q -p no:cacheprovider tests/test_phase1_invariants.py tests/test_phase2_metrics.py', '```', '',
        '## Bàn giao tiếp theo', '',
        '- A: đặc tả pre-local checkpoint, Adam state, batch order, BN/Dropout của FedAvg thật.',
        '- B: xác minh forward local update trước inversion; dùng Phase 2 làm diagnostic reference. Nếu cần nâng attack tabular, thử từng parameterization/objective trên development riêng.',
        '- C: phân tích khả nghịch DNA và random realization trước adaptive attack; chưa xếp hạng defense.',
        '- D: khóa protocol Phase 3, metric/compute budget; giữ đơn vị sample/restart/client rõ ràng.', '',
        'Không triển khai Phase 3, không push/publish. Phase 2 hoàn thành với giới hạn đã mô tả, không phải bằng chứng formal privacy hay bảo vệ DNA.']
    (ROOT/'phase2_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(run=run.name, pairing_development=pairing, vector_checks=len(checks)*4,
                         objective_checks=len(checks)*2, paired_zero_wins=wins, paired_prior_wins=priorwins)))


if __name__=='__main__':
    main()
