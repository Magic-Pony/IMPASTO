"""Build a readable media report from already measured scene/stage-1 results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/scene_audit'


def main():
    a = json.loads((OUT/'metrics.json').read_text())
    s = json.loads((ROOT/'reports/stage1/metrics.json').read_text())
    cfg = json.loads((ROOT/'configs/stage1.json').read_text())
    assert cfg == a['config'], 'Audit must be rerun for the current configuration'
    assert s['minimum_stroke_contact_fraction'] > .99
    assert not a['unexpected_penetrations_over_0_5mm']
    # Side elevation of the local assembly, all coordinates in mm from hand.
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    pieces = [(-25,83,50,24,'#aab1b9','Grasping block: 50 x 36 x 24'),
              (-28,107,56,8,'#7f8e9f','Upper flange: D56 x 8'),
              (-27,115,54,30,'#8e7959','F/T housing: D54 x 30'),
              (-28,145,56,8,'#7f8e9f','Lower flange: D56 x 8'),
              (-10,153,20,24,'#afb6bd','Holder stem: D20 x 24'),
              (-9,169,18,18,'#31363d','Collet: D18 x 18'),
              (-3,171,6,14,'#bb8549',''),
              (-4,185,8,16,'#7f8e9f','Ferrule: 16'),
              (-4,201,8,20,'#292823','Bristles: 20')]
    for x,y,w,h,c,label in pieces:
        ax.add_patch(Rectangle((x,y),w,h,facecolor=c,edgecolor='white',lw=.7))
        if label:
            ty={149:152,165:170,178:187,193:205,211:221}.get(y+h/2,y+h/2)
            ax.annotate(label,(x+w,y+h/2),(42,ty),fontsize=10,va='center',
                        arrowprops={'arrowstyle':'-','color':'#666','lw':.8})
    for x,lo,hi,txt in [(-43,171,221,'Cut brush 50'),(-61,187,221,'Exposed 34')]:
        ax.annotate('',(x,lo),(x,hi),arrowprops={'arrowstyle':'<->','color':'#17679a'})
        ax.text(x-2,(lo+hi)/2,txt,rotation=90,ha='right',va='center',fontsize=10,color='#17679a')
    ax.axhline(171,xmin=.31,xmax=.54,color='#17679a',ls=':',lw=.8)
    ax.set(xlim=(-80,165),ylim=(238,72),ylabel='Distance along tool axis from hand origin (mm)')
    ax.set_xticks([]); ax.spines[['top','right','bottom']].set_visible(False)
    ax.set_title('Estimated local assembly | not paper CAD\n50 mm includes 16 mm inside the collet',loc='left',fontsize=14)
    fig.savefig(OUT/'tool_dimensions.png',dpi=160);plt.close(fig)
    summary = f'''原长杆缺少论文中的安装件，已改为夹持块、上下法兰、F/T 外壳、夹具与短画笔。
论文的“trimmed to 5 cm”按修剪后整笔 50 mm 解释；其中 16 mm 插入夹具，34 mm 外露。
木柄 14 mm、金属箍 16 mm、刷毛 20 mm，传感器外壳约 Φ54×30 mm：这些拆分尺寸均为本地估计。
调色盘与水盆分别放在画布两侧，目标位置从场景几何读取，采用竖直进入与平滑换位。'''
    rows = [
        ('蘸漆 / 清洗实际验证','24 / 24 次，各全部成功'),
        ('工位最大静态 IK 误差',f"{a['static_max_error_m']*1000:.3f} mm"),
        ('浸入时最大 XY 误差',f"{a['dynamic_max_xy_error_m']*1000:.3f} mm"),
        ('整个路径最大刷杆倾角',f"{a['all_movement_max_tilt_deg']:.3f}°"),
        ('浸入时最大刷杆倾角',f"{a['dynamic_max_immersion_tilt_deg']:.4f}°"),
        ('超过 0.5 mm 的意外穿透',str(len(a['unexpected_penetrations_over_0_5mm']))),
        ('单笔 + 多笔绘画',f"{s['success_count']}/{s['all_stroke_count']} 成功，各笔全程接触"),
        ('9 笔平均力 RMSE',f"{s['mean_force_rmse_N']:.3f} N"),
        ('9 笔平均力标准差',f"{s['mean_force_std_N']:.3f} N"),
        ('9 笔平均轨迹 RMSE',f"{s['mean_trajectory_rmse_m']*1000:.2f} mm"),
        ('无录像的仿真 / 墙钟时间比',f"{s['simulation_realtime_factor_without_render']:.2f}×"),
        ('4 进程吞吐 / 串行加速',f"{s['throughput']['parallel_strokes_per_s']:.2f} 笔/s / {s['throughput']['speedup']:.2f}×")]
    limits = '''这些误差来自理想仿真，不代表真实机器人精度。保留 Panda、画布尺寸、24 色槽、顶视观测与附录 A 的工具结构；安装尺寸、质量、工位坐标为本地设计。刷毛使用轴向弹簧近似，颜料为接触驱动沉积模型，水盆无流体仿真。
当前控制仍是关节伺服加连续法向高度反馈，没有完整实现论文式 (4) 的 JᵀF 前馈力矩注入与运笔保持高度策略。原有接触力乘 0.08 的缩放已移除，报告力值是 MuJoCo 未缩放接触力；安装点六维传感器单独保留。
演示是预设笔画，不是学习模型或 MPC 的结果。未验证论文整个 0.1–4 N 工况。物理步长 2 ms，控制周期 4 ms，视频按仿真时间采样，不加速播放。'''
    table = '\n'.join(f'|{k}|{v}|' for k,v in rows)
    md = f'''# IMPASTO：短笔装配与工位布局修正

{summary}

![工具近景](tool_closeup.png)

![尺寸估计](tool_dimensions.png)

## 实测验收

|指标|本轮结果|
|---|---:|
{table}

![蘸漆近景](dip_closeup.png)

![清洗近景](clean_closeup.png)

![力与压缩曲线](force.png)

## 视频

- [8 笔场景视频（含换色与清洗）](../stage1/videos/stage1_scene.mp4)
- [同步顶视视频](../stage1/videos/stage1_topdown.mp4)
- [单笔视频](../stage1/videos/single_stroke_scene.mp4)

## 对齐边界

{limits}

- [逐项论文核对](../../docs/scene_audit.md)
- [工位与单笔原始数值](metrics.json)
- [力曲线 CSV](force_trace.csv)
- [阶段 1 综合数值](../stage1/metrics.json)
- [阶段 1 完整报告](../stage1/report.md)

复查：`bash scripts/run.sh -m pytest -q`，`bash scripts/run.sh scripts/audit_scene.py`，`bash scripts/run.sh scripts/run_stage1.py`，`bash scripts/run.sh scripts/build_scene_report.py`。
'''
    (OUT/'report.md').write_text(md)
    trs=''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k,v in rows)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IMPASTO · 工具与场景修正</title><style>
body{{margin:0;background:#f1f2ee;color:#202d32;font:17px/1.75 system-ui,sans-serif}}main{{max-width:1120px;margin:0 auto;padding:42px 24px}}h1{{font-size:34px;line-height:1.35}}h2{{margin:32px 0 14px}}p{{white-space:pre-line}}.tag{{color:#34786f;font-weight:700;letter-spacing:2px}}.panel{{background:white;padding:24px;border-radius:14px;margin:20px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}img,video{{max-width:100%;display:block;border-radius:6px}}video{{background:#18222a;width:100%}}figure{{margin:0}}figcaption{{font-size:14px;color:#667;padding:8px 0}}table{{border-collapse:collapse;width:100%}}td{{padding:8px 12px;border-bottom:1px solid #ddd}}td:last-child{{text-align:right;font-variant-numeric:tabular-nums}}a{{color:#24677e}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}h1{{font-size:27px}}}}
</style><main><span class="tag">IMPASTO / LOCAL SIMULATION</span><h1>短笔装配与工位布局修正</h1><p>{summary}</p>
<div class="grid"><figure><img src="tool_closeup.png"><figcaption>实际 MuJoCo 工具近景</figcaption></figure><figure><img src="tool_dimensions.png"><figcaption>本地估计尺寸（非作者 CAD）</figcaption></figure></div>
<section class="panel"><h2>实际验收结果</h2><table>{trs}</table></section>
<div class="grid"><figure><img src="dip_closeup.png"><figcaption>笔尖竖直浸入颜料</figcaption></figure><figure><img src="clean_closeup.png"><figcaption>笔尖竖直浸入水面，旋转器运动</figcaption></figure></div>
<h2>8 笔场景录像</h2><video controls preload="metadata" poster="scene.png" src="../stage1/videos/stage1_scene.mp4"></video><p><a href="../stage1/videos/stage1_topdown.mp4">同步顶视视频</a> · <a href="../stage1/videos/single_stroke_scene.mp4">单笔视频</a></p>
<h2>接触力和刷毛压缩</h2><img src="force.png">
<section class="panel"><h2>论文对齐与边界</h2><p>{limits}</p><p><a href="../../docs/scene_audit.md">逐项论文核对</a> · <a href="metrics.json">工位数值</a> · <a href="force_trace.csv">力曲线 CSV</a> · <a href="../stage1/metrics.json">阶段 1 数值</a></p></section></main></html>'''
    (OUT/'index.html').write_text(html)
    print(OUT/'index.html')


if __name__ == '__main__':
    main()
