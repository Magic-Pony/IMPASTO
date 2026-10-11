#!/usr/bin/env python3
"""Build a report only from complete runs, frozen checkpoints and visual notes."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import shutil
import statistics
import imageio.v2 as imageio
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def table(headers, rows):
    cells = lambda tag, row: ''.join(f'<{tag}>{html.escape(str(x))}</{tag}>' for x in row)
    return '<table><thead><tr>' + cells('th', headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + cells('td', row) + '</tr>' for row in rows) + '</tbody></table>'


def picture(src, alt):
    return f'<img loading="lazy" src="{html.escape(src)}" alt="{html.escape(alt)}">'


def video_contact_sheet(out, task):
    """Sample the encoded continuous replay, rather than reconstructed canvases."""
    record = json.loads((out / f'{task}_scene_replay.json').read_text())
    reader = imageio.get_reader(out / f'{task}_scene_replay.mp4')
    try:
        count = reader.count_frames()
        assert count == record['frames']
        indices = [int((count - 1) * fraction) for fraction in (.12, .28, .44, .60, .82, 1.)]
        sheet = Image.new('RGB', (1440, 3 * 300), 'white')
        draw = ImageDraw.Draw(sheet)
        for n, index in enumerate(indices):
            frame = Image.fromarray(reader.get_data(index)).resize((720, 270))
            x, y = (n % 2) * 720, (n // 2) * 300
            draw.text((x + 10, y + 8), f'{task} | frame {index}/{count - 1}', fill='black')
            sheet.paste(frame, (x, y + 30))
        sheet.save(out / f'{task}_video_contact_sheet.png')
    finally:
        reader.close()
    return {'frames': count, 'sampled_frames': indices, 'fps': record['fps'],
            'encoded_duration_seconds': count / record['fps'],
            'final_canvas_max_abs_rgb_difference': record['final_canvas_max_abs_rgb_difference']}


def validation_table(data):
    rows = [
        [key, f'{data["means"]["previous"][key]:.6f}',
         f'{data["means"]["proposed"][key]:.6f}',
         f'{1-data["means"]["proposed"][key]/data["means"]["previous"][key]:.2%}']
        for key in ('weighted_gray_l1', 'changed_region_gray_l1')]
    for key in ('target_foreground_rgb_l1', 'union_foreground_rgb_l1'):
        a, b = [statistics.mean(r[label]['region_metrics'][key] for r in data['samples'])
                for label in ('previous', 'proposed')]
        rows.append([key, f'{a:.6f}', f'{b:.6f}', f'{1-b/a:.2%}'])
    return table(['指标（越低越好）', '1107014', '本轮', '相对降幅'], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', default='reports/refinement_coverage_order')
    parser.add_argument('--development-dir', default='reports/general_validation_coverage_order')
    parser.add_argument('--validation-dir', default='reports/general_validation_fresh')
    args = parser.parse_args()
    out = ROOT / args.report_dir
    current = read(f'{args.report_dir}/metrics.json')
    audit = read(f'{args.report_dir}/audit/visual_audit_metrics.json')
    notes = read(f'{args.report_dir}/visual_notes.json')
    previous = read('reports/stage3_rollback_1107014/metrics.json')
    v4 = read('reports/refinement_final_v4/metrics.json')
    calibrated = read('reports/refinement_calibrated/metrics.json')
    force_old = read('reports/force_model_stable_old/metrics.json')['summary']
    force_raw = read('reports/force_model_stable_new/metrics.json')['summary']
    force_calibrated = read('reports/force_model_calibrated/metrics.json')['summary']
    model = read('reports/stage2_stable/evaluation_metrics.json')
    train = read('reports/stage2_stable/training_metrics.json')
    calibration = read('reports/model_calibration_audit/metrics.json')
    overlap = read('reports/chromatic_overlap_audit/metrics.json')
    development = read(f'{args.development_dir}/metrics.json')
    validation = read(f'{args.validation_dir}/metrics.json')
    checkpoint = current['paper_protocol']['checkpoint_sha256']
    assert train['epochs'] == 1000
    assert checkpoint == calibration['variants']['best_calibrated']['sha256']
    assert checkpoint == validation['models']['proposed']['sha256']
    assert len(development['samples']) == len(validation['samples']) == 8
    assert validation['seed_start'] != development['seed_start']
    video_checks = {}
    for task in ('flower', 'fish'):
        assert len(notes['rounds'][task]) == len(current['experiments'][task]['cycles'])
        assert read(f'{args.report_dir}/{task}_scene_replay.json')['final_canvas_max_abs_rgb_difference'] == 0
        video_checks[task] = video_contact_sheet(out, task)
    (out / 'video_review.json').write_text(json.dumps(video_checks, indent=2) + '\n')
    rows = []
    for task in ('flower', 'fish'):
        a = previous['experiments'][task]['metrics']['weighted_gray_l1']
        b = v4['experiments'][task]['metrics']['weighted_gray_l1']
        d = calibrated['experiments'][task]['metrics']['weighted_gray_l1']
        c = current['experiments'][task]['metrics']['weighted_gray_l1']
        reg = audit['experiments'][task]['final']
        rows.append([task, f'{a:.6f}', f'{b:.6f}', f'{d:.6f}', f'{c:.6f}', f'{1-c/a:.2%}',
                     f'{reg["foreground_iou"]:.3f}', f'{reg["target_foreground_recall"]:.3f}',
                     f'{reg["union_foreground_rgb_l1"]:.4f}'])
    region_rows = []
    for task in ('flower', 'fish'):
        record = audit['experiments'][task]
        a, b = record['previous_on_current_target'], record['final']
        region_rows.append([task, f'{a["foreground_iou"]:.4f}', f'{b["foreground_iou"]:.4f}',
            f'{a["target_foreground_recall"]:.4f}', f'{b["target_foreground_recall"]:.4f}',
            f'{a["union_foreground_rgb_l1"]:.5f}', f'{b["union_foreground_rgb_l1"]:.5f}'])
    control_rows = []
    for label, s in [('旧物理 / 旧模型', force_old['original']),
                     ('稳定物理 / 旧模型', force_old['stable']),
                     ('稳定物理 / 新原始模型', force_raw['stable_raw']),
                     ('稳定物理 / 新校准模型', force_calibrated['stable_raw']),
                     ('稳定物理 / 新校准模型 + 可行性投影', force_calibrated['stable'])]:
        control_rows.append([label, f'{s["mean_force_all_samples_rmse_N"]:.5f}',
            f'{s["worst_force_all_samples_rmse_N"]:.5f}', f'{s["minimum_contact_fraction"]:.5f}',
            f'{s["mean_prediction_union_gray_l1"]:.5f}'])
    changes = [
        ['选择性回退', '保留 1107014 的对称灰度目标、RGB 权重 1、原宽度/沉积律；弃用 force cap 和过度压窄约束；统一使用全部 24 色。', '本地版本选择；删除按任务限制颜色的做法'],
        ['阶段一', 'MuJoCo 刷尖接触松弛时间 6→20ms；2ms timestep、刷毛弹簧和持续力反馈不变。随机动作证明原低力振荡。', '仿真数值修复；不是论文参数，也没有证明论文控制公式有误'],
        ['阶段二', '2000 条随机 self-play，1800/200 按 episode 划分；提高低压力采样比例；拒收断触/大力误差。保持 UNet、Adam、batch120、1000epochs 和学习率衰减。', '通用扩展；超过论文 900/100 的数据预算'],
        ['BatchNorm', '按网络前向顺序，仅用未增强训练集重估 BN 总体统计；关闭 dropout，不改权重、不做梯度更新。', '通用部署校准；论文未规定；原始新模型并非全面改善'],
        ['候选提取', '保留细线连通区域；RGB 已识别区域不再被第二次灰度门限丢弃；PCA/RANSAC 增加颜色区域和多力假设；用 learned dynamics 排序。', '本地预处理修复和通用候选扩展'],
        ['颜色采样', '足迹与中心圆盘交集、距离加权、线性 RGB 均值、Mahalanobis 分类；中心圆盘半径统一 15mm。', '更对齐 Appendix B；15mm 是本地数值；72组随机分类测试两种方法均100%'],
        ['颜料可行性', '预测灰度限制在当前灰度与颜料灰度之间，防止虚构擦除或亮度越界；空间变化仍来自学习模型。', '依据本地凸混合沉积不变量的通用扩展；不是论文原算法'],
        ['等灰度覆盖', '用同一学习模型对空白参考做反事实预测，估计覆盖率，补足灰度近似相同、色相不同的换色区域。', '通用 RGB 预测修复；无解析真值、无目标动作输入'],
        ['顺序搜索', 'H≤5 时枚举最多120个种子排列，用同一模型和目标代价选覆盖顺序，再正常进行 MPPI。', '通用规划扩展；论文未明确采用'],
        ['运行速度与追溯', '预渲染 disk stamp 改为最近采样点查询，200随机动作逐像素相同，局部函数快3.10倍；保存配置、源码和模型哈希。', '工程优化；没有减少规划候选或迭代预算']]
    body = '<h1>IMPASTO：选择性回退与通用修正</h1><p class="lead">' + html.escape(notes['conclusion']) + '</p>'
    body += '<p>所有目标共享配置、24 色调色盘和动作空间。Flower/Fish 是测试图片；没有按画名、坐标、轮次或指定笔迹插入动作。保留 learned dynamics、PCA/RANSAC、H=5、执行最多两笔后读取真实仿真画布重规划，以及论文第5/10/15/final检查点。未复现论文 baselines。</p>'
    body += '<h2>最终画面与 1107014 对照</h2>' + table(['任务', '1107014 L1', '候选修正 v4', '校准模型对照', '本轮 L1', '相对降幅', '前景 IoU', '前景召回', '前景并集 RGB L1'], rows)
    body += '<h3>形状覆盖与RGB误差的取舍</h3>' + table(['任务', '旧 IoU', '本轮 IoU', '旧召回', '本轮召回', '旧 RGB L1', '本轮 RGB L1'], region_rows)
    body += '<p>Flower覆盖率略升，Fish覆盖率明显退化：IoU 0.7279→0.6208，召回0.7664→0.6351。Fish虽然灰度和RGB平均误差下降，却漏了更多目标区域，视觉不能称全面改善。前景按各通道与纸面颜色最大差&gt;15/255定义，极淡笔迹可能未计入；图片和逐轮观察仍是必要证据。</p>'
    body += '<p>校准模型对照采用相同稳定物理、数据和候选设计，关闭等灰度覆盖率修复及种子顺序搜索。该对照与本轮都完整执行；两项新增规划功能的贡献作为一个组合比较，不能推断各自的独立贡献。</p>'
    body += '<p>加权灰度 L1 在相同公开 Fig.8 目标上计算，越低越好；MPC 综合代价另含 RGB 辅助项。原高清目标、逐笔动作、硬件数据未公开，本地目标来自图版裁剪，不能把这些数值与论文机器人性能等同。LPIPS 未计算。</p>'
    for task in ('flower', 'fish'):
        body += f'<h3>{task.title()}</h3>' + picture(f'audit/{task}_comparison.png', '目标、1107014与本轮实际执行')
    single = current['experiments']['single_stroke']
    body += '<h2>单笔闭环检查</h2>' + table(['项', '结果'], [
        ['1107014 实际执行加权灰度 L1', f'{previous["experiments"]["single_stroke"]["metrics"]["weighted_gray_l1"]:.6f}'],
        ['实际执行加权灰度 L1', f'{single["metrics"]["weighted_gray_l1"]:.6f}'],
        ['执行成功', single['execution_success']],
        ['法向力 RMSE / N', f'{single["execution_force_rmse_mean_N"]:.6f}'],
        ['轨迹 RMSE / mm', f'{1000*single["execution_trajectory_rmse_mean_m"]:.4f}']])
    body += '<p>本轮单笔误差较1107014升高20.65%，属于退化。单笔目标由本地仿真生成，规划器只接收图像，没有得到用于生成目标的动作；它验证本地闭环，不能代替公开艺术目标测试。下图为目标、实际执行及MPC灰度预测。</p>'
    body += picture('single_result.png', '单笔目标、执行与预测')
    body += '<h2>整幅画执行质量</h2>' + table(['任务', '实际笔数', '重规划轮数', '力 RMSE / N', '轨迹 RMSE / mm', '运行耗时 / s'], [
        [task, current['experiments'][task]['executed_strokes'],
         current['experiments'][task]['replanning_cycles'],
         f'{current["experiments"][task]["execution_force_rmse_mean_N"]:.5f}',
         f'{1000*current["experiments"][task]["execution_trajectory_rmse_mean_m"]:.4f}',
         f'{current["experiments"][task]["wall_seconds"]:.1f}']
        for task in ('flower', 'fish')])
    body += '<h2>修改内容与论文边界</h2>' + table(['项', '修改', '归类'], changes)
    body += '<p>本轮没有认定论文公式错误。MuJoCo 接触、理想力信号及接触驱动的局部油漆混合属于仿真替代；没有刷毛分辨形变、油漆流体或真实力传感器链。恒力、恒色二次曲线的表达能力也限制了照片中的变宽、尖端和浓淡。</p>'
    body += '<h2>力控与局部预测：分别核查</h2>' + table(['物理 / 模型', '平均力 RMSE / N', '最差力 RMSE / N', '最低接触比例', '局部预测灰度 L1'], control_rows)
    body += '<p>两条独立曲线×7个力值，包含空白与交叉背景，动作和种子相同。稳定接触显著降低力误差，但新校准模型的局部预测误差仍高于旧模型；不能称模型全面优于旧模型。局部误差在预测与真实变化区域并集计算。</p>'
    body += picture('../force_model_calibrated/thin_stroke_predictions.png', '校准模型在各力值的实际笔迹与预测')
    body += '<h2>2000条数采、完整训练与统计校准</h2>'
    body += table(['同一200条验证集', '全 patch L1', '真实变化区域 L1', '预测/真实并集 L1'], [
        [name, f'{rec["full_patch_l1"]:.6f}', f'{rec["changed_region_l1"]:.6f}', f'{rec["union_region_l1"]:.6f}']
        for name, rec in calibration['variants'].items()])
    q = model['collection_quality']
    body += f'<p>20个episode各100笔，1800训练/200验证；24色随机曲线，实际低力≤0.65N占59.1%。平均力RMSE {q["mean_force_rmse_N"]:.5f}N、最大 {q["max_force_rmse_N"]:.5f}N、最低接触比例 {q["min_contact_fraction"]:.3f}，拒收95次后补采。完整1000epochs，最佳原始checkpoint在第{train["best_epoch"]}epoch，训练耗时{train["training_wall_seconds"]:.1f}s。原始新模型全图误差降低而笔迹区域误差变差，不能仅看全图均值。校准只接触训练集，但上述验证集参与了checkpoint与版本选择，属于开发验证。</p>'
    body += picture('../stage2_stable/training_curves.png', '完整训练曲线')
    body += '<h2>等灰度换色的独立功能检查</h2>' + table(['8组随机交叉、相同灰度预测', '旧RGB转换', '增加学习覆盖率'], [
        [key, f'{overlap["means"]["previous"][key]:.6f}', f'{overlap["means"]["proposed"][key]:.6f}']
        for key in ('changed_region_rgb_l1', 'ambiguous_region_rgb_l1')])
    body += '<p>选择灰度接近而RGB不同的颜料对，随机交叉曲线，不使用Flower/Fish。覆盖率来自同一网络对空白参考的预测。旧转换在交叉处保留底色或断开新笔迹；新转换恢复覆盖，形状边缘仍有误差。这是功能修复证据，不能等同任意整幅画都改善。</p>'
    body += picture('../chromatic_overlap_audit/comparison.png', '8组随机等灰度覆盖预测对照')
    body += '<h2>通用随机目标：开发对照与新种子测试</h2>'
    body += '<p>规划器只看目标图像，不获取生成目标的动作。旧/新配置分别加载匹配的物理与模型；统一24色，32候选、16迭代，两笔后闭环。新增种子排序和覆盖率查询带来额外模型评估，所以这不是相同总计算量的对照。目标由稳定物理配置下完整MuJoCo执行产生，测试物理/模型/规划的组合；不能分离各项贡献或据此保证所有艺术照片。</p>'
    body += f'<h3>开发集：8个三笔目标</h3>{validation_table(development)}<p>{development["wins"]}/8加权灰度改善。这组目标用于选择模型/方案，不能再称未参与调试的测试集。</p>'
    body += picture(f'../{Path(args.development_dir).name}/comparison.png', '开发随机目标对照')
    body += f'<h3>新种子：8个四笔目标</h3>{validation_table(validation)}<p>种子从{validation["seed_start"]}开始，方案冻结后运行，未用结果继续调参。{validation["wins"]}/8加权灰度改善。{html.escape(notes["generalization"])}</p>'
    body += picture(f'../{Path(args.validation_dir).name}/comparison.png', '新种子四笔目标对照')
    body += '<h2>用户现象的核查与归因</h2>' + table(['现象', '判断'], notes['observations'].items())
    body += '<h2>逐轮视觉审计</h2><p>八图依次为目标、规划前观测、PCA/RANSAC区域与种子中心线、优化动作、H笔灰度终态、此次执行前缀预测、同样前缀真实执行、误差。掩码颜色表示区域编号，不是颜料。另提供RGB终态/前缀对照及逐笔预测，区分目标不匹配、模型预测误差和执行误差。已逐轮人工视觉检查。</p>'
    for task in ('flower', 'fish'):
        body += f'<h3>{task.title()}</h3>'
        for i, (note, rec) in enumerate(zip(notes['rounds'][task], audit['experiments'][task]['cycles']), 1):
            body += f'<details><summary>第{i}轮：{rec["before"]}→{rec["after"]}笔；预测/执行局部灰度L1 {rec["prediction_execution_union_gray_l1"]:.4f}，RGB L1 {rec["prediction_execution_union_rgb_l1"]:.4f}</summary><p>{html.escape(note)}</p>'
            for kind, alt in [('audit', '八图诊断'), ('rgb_audit', 'RGB预测/执行对照'), ('rgb_rollout', '每笔RGB预测')]:
                body += picture(f'audit/{task}_{kind}_{i:02d}.png', f'{task}第{i}轮{alt}')
            body += '</details>'
    body += '<h2>连续真实仿真重放</h2><p>保存的最终动作重新执行，视频包含清洗、取色、转场、接触、运笔和实时画布，按仿真时间采样；不是静态画布拼帧。两画重放与规划执行最终RGB逐像素一致，最大差0。视频不包含求解器计算等待。</p>'
    for task in ('flower', 'fish'):
        body += f'<h3>{task.title()}</h3><video controls preload="metadata" src="{task}_scene_replay.mp4"></video>'
        body += picture(f'{task}_video_contact_sheet.png', '人工核查的连续场景视频抽帧')
    body += '<p>' + html.escape(notes['video_review']) + '</p>'
    body += '<h2>失败尝试与剩余限制</h2>'
    body += '<p>强窄笔/force cap使Flower L1退化到0.01671；1ms/0.5ms步长未完全消除断触，未采用。候选修正v4在8个开发目标上均值仅改善0.47%；薄线/RGB区域/密集颜色采样单独配旧模型反而退化5.1%；加可行性投影仅改善1.36%。这些改动本身不足以解决问题。旧模型BN校准和新模型最后epoch校准均未胜过最终候选。新原始模型完整绘画实验因局部预测退化提前终止，refinement_stable是未完成尝试，不能当作最终结果。</p>'
    for note in notes['limitations']:
        body += '<p>' + html.escape(note) + '</p>'
    body += '<h2>完整证据与复现入口</h2><p>' + ' · '.join(
        f'<a href="{path}">{label}</a>' for path, label in [
            ('metrics.json', '最终动作与数值'), ('resolved_config.json', '冻结配置'),
            ('audit/visual_audit_metrics.json', '逐轮数值'), ('visual_notes.json', '逐轮视觉归因'),
            ('video_review.json', '连续视频核查'),
            ('../stage2_stable/dataset_manifest.json', '数采清单'),
            ('../model_calibration_audit/metrics.json', '校准版本对照'),
            (f'../{Path(args.validation_dir).name}/metrics.json', '新种子测试记录')]) + '</p>'
    body += f'<p>Checkpoint SHA-256：<code>{checkpoint}</code>。数据和模型保留在data/stage2_stable、outputs/stage2_stable，由Git忽略；代码、配置、清单、数值、图文报告和场景视频进入Git。完整运行命令见docs/general_refinement.md。</p>'
    css = 'body{max-width:1180px;margin:36px auto;padding:0 24px;font:17px/1.7 system-ui;background:#f5f5f2;color:#24302e}h1,h2,h3{line-height:1.3}h2{margin-top:48px}.lead{font-size:21px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:10px;text-align:left;border-bottom:1px solid #ccd3ce}th{background:#e2e8df}img,video{display:block;max-width:100%;height:auto;margin:20px auto;background:white}details{padding:15px;margin:14px 0;background:white;border:1px solid #d7ddd8;border-radius:8px}summary{cursor:pointer;font-weight:600}code{overflow-wrap:anywhere}a{color:#256257}'
    (out / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>IMPASTO 通用修正与逐轮审计</title><style>' + css + '</style><body>' + body + '</body></html>')
    generated = out / 'report.md'
    if generated.exists() and not (out / 'experiment_report.md').exists():
        shutil.copy2(generated, out / 'experiment_report.md')
    md = ['# IMPASTO：选择性回退与通用修正', '', notes['conclusion'], '',
          '[完整图片、逐轮RGB审计与连续视频](index.html)', '',
          '|任务|1107014 L1|候选v4 L1|校准对照L1|本轮L1|相对降幅|IoU|召回|RGB L1|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    md += ['|' + '|'.join(row) + '|' for row in rows]
    md += ['', '|任务|旧IoU|本轮IoU|旧召回|本轮召回|旧RGB L1|本轮RGB L1|',
           '|---|---:|---:|---:|---:|---:|---:|']
    md += ['|' + '|'.join(row) + '|' for row in region_rows]
    md += ['', 'Fish平均误差下降，但IoU和召回明显退化，不能称视觉全面改善。单笔闭环L1从0.0009305升至0.0011227，退化20.65%。']
    md += ['', '## 修改与归类', ''] + [f'- **{a}**：{b} {c}。' for a, b, c in changes]
    for label, data in [('开发集（三笔）', development), ('新种子测试（四笔）', validation)]:
        md += ['', f'## {label}', '', f'{data["wins"]}/8加权灰度误差改善。', '']
        for key in ('weighted_gray_l1', 'changed_region_gray_l1'):
            a, b = data['means']['previous'][key], data['means']['proposed'][key]
            md += [f'- {key}：{a:.6f} → {b:.6f}，下降 {1-b/a:.2%}。']
    md += ['', notes['generalization'], '', '## 用户现象的核查', '']
    md += [f'- **{k}**：{v}' for k, v in notes['observations'].items()]
    for task in ('flower', 'fish'):
        md += ['', f'## {task.title()}逐轮视觉审计', '']
        for i, note in enumerate(notes['rounds'][task], 1):
            md += [f'{i}. {note}', f'\n   [八图](audit/{task}_audit_{i:02d}.png) · [RGB逐笔预测](audit/{task}_rgb_rollout_{i:02d}.png)\n']
        md += ['', f'[连续仿真视频]({task}_scene_replay.mp4)', f'\n![最终对照](audit/{task}_comparison.png)']
    md += ['', '## 剩余限制', ''] + [item for note in notes['limitations'] for item in (note, '')] + ['',
        f'完整1000epochs训练；校准checkpoint SHA-256：`{checkpoint}`。', '',
        '全图均值、局部模型误差、失败试验与校准版本详见HTML。未声称达到论文硬件效果或证明任意艺术目标均可复现。']
    generated.write_text('\n'.join(md) + '\n')
    print(out / 'index.html')


if __name__ == '__main__':
    main()
