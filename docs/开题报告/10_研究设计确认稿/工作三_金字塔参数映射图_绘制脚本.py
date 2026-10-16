from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch

BASE=Path(__file__).resolve().parent
FONT=FontProperties(fname='/workspace/scratch/0dffb7d03718/report_revision_20261010_1020/qa_fonts/DroidSansFallback.ttf')
plt.rcParams['svg.fonttype']='path'
W,H=1800,1120
fig=plt.figure(figsize=(18,11.2),dpi=100,facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off')
audit=[]
def text(x,y,s,size=14,ha='left',va='top',bound=None):
 t=ax.text(x,y,s,fontproperties=FONT,fontsize=size,color='black',ha=ha,va=va)
 if bound: audit.append((s,t,bound))
 return t
def box(x,y,w,h,title,lines,size=14):
 ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='black',linewidth=1.35))
 text(x+17,y+13,title,17,bound=(x+7,y+7,x+w-7,y+h-7))
 ax.plot([x+14,x+w-14],[y+48,y+48],color='black',linewidth=.65)
 for i,s in enumerate(lines): text(x+17,y+61+i*29,s,size,bound=(x+7,y+7,x+w-7,y+h-7))
def arrow(pts,dashed=False):
 ls=(0,(4,4)) if dashed else 'solid'
 if len(pts)>2:
  xs,ys=zip(*pts[:-1]);ax.plot(xs,ys,color='black',linewidth=1.25,linestyle=ls)
 ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=14,linewidth=1.25,linestyle=ls,color='black',shrinkA=0,shrinkB=0))

text(50,25,'生产信息层级与可训练融合模块',25)
text(50,77,'金字塔表示生产信息范围；右侧为独立参数分组候选。更新频率须验证，尚不代表代码已实施。',14)

# Pyramid: physical hierarchy, not a declaration of GNN depth.
cx,tip,base_y,half=425,170,845,375
def edges(y):
 w=half*(y-tip)/(base_y-tip)
 return cx-w,cx+w
cuts=[tip,370,605,base_y]
for a,b in zip(cuts[:-1],cuts[1:]):
 la,ra=edges(a);lb,rb=edges(b)
 ax.add_patch(Polygon([(la,a),(ra,a),(rb,b),(lb,b)],closed=True,facecolor='white',edgecolor='black',linewidth=1.5))
text(cx,289,'整线',22,ha='center')
text(cx,332,'各站／周期／批次',12.2,ha='center')
text(cx,454,'站位',22,ha='center')
text(cx,501,'当前／下一周期任务',13.5,ha='center')
text(cx,536,'本站人员与已知资源',13.5,ha='center')
text(cx,681,'工序／飞机',22,ha='center')
text(cx,734,'工序状态与工艺、人员关系',14,ha='center')
text(cx,770,'当前／下一／更后未完任务',14,ha='center')

box(880,129,860,124,'转站与站位就绪时间预测辅助任务  ψ',[
 '当前整线时间修正量＋下一周期各站就绪偏差。',
 '实际结果形成辅助损失，反向训练共享图编码与分层融合。'],13.8)
box(880,284,860,165,'整线融合模块  θ_line',[
 '各站当前／下一周期表示＋实际周期位置与批次字段 → 整线向量。',
 '五站可以按物理顺序拼接；各站表示也保留供候选读取。',
 '参数更新间隔 K_line：训练方案候选，数值与快慢关系待验证。'],13.8)
box(880,495,860,165,'站位融合模块  θ_station',[
 '当前飞机任务／下一架任务＋本站人员与站位表示 → 站位向量。',
 '当前与下一周期分别保留；各站共享模块参数，角色进入输入。',
 '参数更新间隔 K_station：控制本参数组的全部训练更新。'],13.8)
box(880,710,860,165,'飞机融合模块  θ_aircraft',[
 '工序向量按飞机和周期角色读出，再融合任务数量、工作量等统计。',
 '当前／下一／更后任务分通道；各架飞机共享模块参数。',
 '参数更新间隔 K_aircraft：无参数池化与可训练MLP分开。'],13.8)
box(50,924,750,122,'复用异构图编码  θ_graph',[
 '完整工序、站位、工人、技能及关系 → 节点向量。',
 '飞机身份用于分组，产线人员不按飞机复制。'],13.6)
box(880,924,860,122,'Actor：候选工序＋各层表示＋固定副本预测数值',[
 '逐项选择工序、站位、人员团队及尽早／对齐。',
 '动作头由PPO训练；图编码和融合同时接受策略与时间辅助损失。'],13.6)
arrow([(425,924),(425,845)])
for y in [347,554,775]: arrow([(edges(y)[1],y),(880,y)])
arrow([(1310,710),(1310,660)])
arrow([(1310,495),(1310,449)])
arrow([(1310,284),(1310,253)])
# All levels retain direct candidate access; the chart does not depict a single top-vector bottleneck.
arrow([(880,855),(845,855),(845,980),(880,980)])
# Dashed auxiliary gradient paths to trainable groups. Parameter gates are applied to all losses.
ax.plot([1767,1767],[185,1061],color='black',linewidth=1.2,linestyle=(0,(4,4)))
arrow([(1740,185),(1767,185)],dashed=True)
for y in [365,578,797]: arrow([(1767,y),(1740,y)],dashed=True)
arrow([(1767,1061),(838,1061),(838,982),(800,982)],dashed=True)
text(50,1080,'实线：信息读出与使用；虚线：时间监督反向训练。K均指训练批次间隔；固定副本仍按原方案独立复制、轨迹内不训练。',12.8)

fig.canvas.draw();renderer=fig.canvas.get_renderer()
errs=[]
for s,t,b in audit:
 bb=t.get_window_extent(renderer)
 p=ax.transData.inverted().transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
 lx,rx=sorted(p[:,0]);ty,by=sorted(p[:,1]);x0,y0,x1,y1=b
 if lx<x0-.5 or rx>x1+.5 or ty<y0-.5 or by>y1+.5: errs.append({'text':s,'extent':[float(lx),float(ty),float(rx),float(by)],'allowed':b})
if errs: raise RuntimeError(json.dumps(errs,ensure_ascii=False))
stem='工作三_生产信息金字塔与可训练模块_候选图_20261012'
fig.savefig(BASE/(stem+'.svg'),facecolor='white',metadata={'Title':stem,'Description':'方案讨论：模块参数分组与不同训练频率仍待确认；非实现或性能证据。'})
fig.savefig(BASE/(stem+'.png'),dpi=150,facecolor='white')
print(json.dumps({'text_boxes_checked':len(audit),'status':'passed','files':[stem+'.svg',stem+'.png']},ensure_ascii=False))
