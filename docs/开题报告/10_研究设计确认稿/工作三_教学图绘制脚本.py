from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.font_manager import FontProperties

BASE = Path(__file__).resolve().parent
CJK = '/workspace/scratch/0dffb7d03718/report_revision_20261010_1020/qa_fonts/DroidSansFallback.ttf'
FP = FontProperties(fname=CJK)
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams['axes.unicode_minus'] = False

class Drawing:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w/100, h/100), dpi=100, facecolor='white')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0,w); self.ax.set_ylim(h,0); self.ax.axis('off')
        self.audit = []
    def text(self, x,y,text,size=14,ha='left',va='top',bound=None):
        t=self.ax.text(x,y,text,fontproperties=FP,fontsize=size,ha=ha,va=va,color='black',linespacing=1.38)
        if bound is not None: self.audit.append((text,t,bound))
        return t
    def box(self,x,y,w,h,title,lines,size=14,title_size=17,step=29):
        self.ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='black',linewidth=1.3))
        self.text(x+17,y+13,title,title_size,bound=(x+8,y+6,x+w-8,y+h-6))
        self.ax.plot([x+14,x+w-14],[y+48,y+48],color='black',linewidth=.7)
        for i,s in enumerate(lines): self.text(x+17,y+61+i*step,s,size,bound=(x+8,y+6,x+w-8,y+h-6))
    def arrow(self,pts,dashed=False):
        ls=(0,(4,4)) if dashed else 'solid'
        if len(pts)>2:
            xs,ys=zip(*pts[:-1]); self.ax.plot(xs,ys,color='black',linewidth=1.2,linestyle=ls)
        self.ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=14,
                                         linewidth=1.2,linestyle=ls,color='black',shrinkA=0,shrinkB=0))
    def label(self,x,y,text,size=12):
        self.ax.text(x,y,text,fontproperties=FP,fontsize=size,ha='center',va='center',
                     bbox=dict(facecolor='white',edgecolor='none',pad=2))
    def save(self,stem):
        self.fig.canvas.draw(); renderer=self.fig.canvas.get_renderer()
        errs=[]
        for s,t,b in self.audit:
            bb=t.get_window_extent(renderer)
            p=self.ax.transData.inverted().transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
            lx,rx=sorted(p[:,0]); ty,by=sorted(p[:,1]); x0,y0,x1,y1=b
            if lx<x0-.5 or rx>x1+.5 or ty<y0-.5 or by>y1+.5:
                errs.append({'text':s,'extent':[float(lx),float(ty),float(rx),float(by)],'allowed':b})
        if errs: raise RuntimeError(json.dumps(errs,ensure_ascii=False))
        self.fig.savefig(BASE/(stem+'.svg'),facecolor='white',metadata={'Title':stem,'Description':'教学示意；非工厂实测或已完成实验结果。'})
        self.fig.savefig(BASE/(stem+'.png'),dpi=150,facecolor='white')
        plt.close(self.fig)
        return {'stem':stem,'text_boxes_checked':len(self.audit),'status':'passed'}

def hierarchy():
    d=Drawing(1600,1130)
    d.text(50,25,'图1  分层信息、事件刷新与训练参数更新',24)
    d.text(50,73,'三个生产层级用于组织信息；现场状态、网络表示和参数训练分别处理。',14)
    d.box(50,116,1500,92,'现场输入：最新可见状态＋最近事件／正式决定',
          ['完整工艺与资源图、飞机位置、有效安排、人员占用和已揭示的开工限制；不读取未来扰动。'],size=13,title_size=17)
    d.box(50,240,985,93,'完整图编码',
          ['复用工序、站位、工人、技能及其关系编码；完整图编码与未完任务汇总分开。'],size=14)
    d.box(50,378,985,123,'工序／飞机层',
          ['读取工序表示；按飞机分为当前、紧邻下一及更后未完任务。',
           '保留工序执行状态、任务数量和已有工作量统计，形成飞机摘要。'],size=14)
    d.box(50,552,985,123,'站位层',
          ['融合本站当前飞机任务、下一次进入本站的飞机任务，以及本站人员状态。',
           '例如：三号站当前是飞机C，下一周期接收当前二号站的飞机D。'],size=14)
    d.box(50,726,985,123,'整线层',
          ['汇总各站状态、实际周期位置、在制数量和待投产数量，形成整线背景。',
           '各层表示也供候选评分读取；并非只保留一个整线向量。'],size=14)
    d.box(50,900,595,137,'Actor动作头',
          ['工序 → 站位 → 具体人员团队 → 尽早／对齐',
           '结合候选表示、各层上下文和外部预测数值。'],size=13,title_size=16)
    d.box(685,900,350,137,'在线时间预测头',
          ['当前整线时间修正',
           '下一周期各站就绪偏差'],size=13,title_size=16)
    for a,b in [(208,240),(333,378),(501,552),(675,726)]: d.arrow([(542,a),(542,b)])
    d.arrow([(542,849),(542,878),(347,878),(347,900)])
    d.arrow([(542,878),(860,878),(860,900)])
    d.box(1100,240,450,187,'事件刷新现场事实',
          ['完工、扰动、正式安排、实际转站等。',
           '例如工序完工：更新该任务与人员，',
           '再刷新飞机、本站和整线信息。',
           '事件发生不等于重新训练权重。'],size=13.2,title_size=16,step=29)
    d.arrow([(1325,240),(1325,208)])
    d.box(1100,464,450,122,'每次完整决策前',
          ['用最新一致状态重新计算图表示、',
           '分层融合及固定副本的预测输出。'],size=13.2,title_size=16)
    d.box(1100,628,450,224,'训练时才更新参数',
          ['PPO：在线Actor及共享编码／融合。',
           '真实时间标签：在线预测头和共享表示。',
           '固定预测副本：同一轨迹内不训练。',
           '飞机／站位／整线权重异频训练：',
           '尚未确定，不能按已实现方案展示。'],size=12.8,title_size=16,step=29)
    d.arrow([(1100,720),(1070,720),(1070,285),(1035,285)],dashed=True)
    d.arrow([(1550,800),(1575,800),(1575,1065),(860,1065),(860,1037)],dashed=True)
    d.arrow([(860,1065),(347,1065),(347,1037)],dashed=True)
    d.label(347,1054,'PPO',11.5)
    d.label(860,1054,'时间监督',11.5)
    d.box(1100,900,450,137,'图中线条',
          ['实线：信息读出、融合和使用。',
           '虚线：训练阶段的参数更新。'],size=13.2,title_size=16)
    d.text(50,1094,'决策使用的显式时间数值来自独立固定副本；本图侧重层级与更新，副本路径在图2展开。',12.7)
    return d.save('工作三_分层信息与更新_教学图_20261012')

def example():
    d=Drawing(1600,1440)
    d.text(50,25,'图2  一次扰动如何进入分层信息、预测与实际时间监督',23)
    d.text(50,74,'数值为教学示意，非工厂实测；预测偏差按时间单位展示，训练标签除以基准节拍10。',13.5)
    d.box(50,112,1500,113,'4时发生事件：二号站飞机D的未开工工序a，开工下界由5推迟至10',
          ['两支合格团队的作业时长分别为3和4；人员均可在10时执行。未来扰动在此刻仍不可知。',
           '当前周期起点为0，固定基准节拍为10；保存本次决策前的现场快照。'],size=13.7,title_size=17,step=26)
    d.box(50,279,460,215,'工序／飞机信息',
          ['D.a：未开工，新开工下界为10。',
           '飞机D当前任务：a。',
           '下一站任务：b、c；标准时长6＋4。',
           '更后任务：保留已知未完任务摘要。'],size=13.3,title_size=16,step=31)
    d.box(570,279,460,215,'站位信息',
          ['二号站：当前D，下一周期接收E。',
           '三号站：当前C，下一周期接收D。',
           '三号站下一任务含D.b、D.c。',
           '人员属于站位，读取技能与占用。'],size=13.3,title_size=16,step=31)
    d.box(1090,279,460,215,'整线信息',
          ['各站启发式就绪估计：',
           '[8，13，10，7，9]，最大值13。',
           '二号站估计使用时长3的合格团队。',
           '在制5架，待投产1架；当前时刻4。'],size=13.3,title_size=16,step=31)
    for x in [280,800,1320]: d.arrow([(x,494),(x,520)])
    d.ax.plot([280,1320],[520,520],color='black',linewidth=1.2)
    d.arrow([(410,520),(410,587)]);d.arrow([(1190,520),(1190,587)])
    d.label(800,546,'按同一组可见事实组织输入；两条网络分别编码、读出、融合。',13)
    d.box(50,587,720,155,'在线Actor路径',
          ['自己的完整图编码 → 分层读出 → MLP融合。',
           '候选评分读取工序、飞机、源站、目标站与整线信息。',
           '附在线时间预测头；PPO与实际时间监督训练。'],size=13.4,title_size=17,step=29)
    d.box(830,587,720,155,'固定预测副本路径',
          ['独立复制编码、读出、MLP和两个预测头。',
           '同一完整轨迹参数固定，每次决策输入最新现场。',
           '示例输出：当前修正＋1；下一偏差[－1，0，＋3，－2，0]。'],size=13.1,title_size=17,step=29)
    d.arrow([(410,742),(410,800)]);d.arrow([(1190,742),(1190,800)])
    d.box(50,800,720,131,'Actor的完整执行动作（示例）',
          ['选D.a → 留二号站 → 选原团队（时长4）→ 尽早。',
           '规则安排10开工、预计14完工；正式提交后锁定选择。'],size=13.5,title_size=17)
    d.box(830,800,720,131,'这些预测数值是什么意思',
          ['当前：启发式13＋修正1，得到预测转站14。',
           '下一：三号站预计晚于基准就绪3；并非硬容量保证。'],size=13.5,title_size=17)
    d.arrow([(830,869),(770,869)])
    d.label(800,772,'预测数值供动作评分',12)
    d.arrow([(410,931),(410,980)])
    d.box(50,980,1500,174,'随后实际执行：事后结果只用于训练，不能进入4时的输入',
          ['13时才揭示D.b的新开工下界18；本次实际转站Pq＝14，下一周期实际起点为14。',
           'D进入三号站后，b在18—24、c在24—28执行；下一周期基准就绪参照为14＋10＝24。',
           '下一周期各站实际就绪为[23，24，28，22，24]，共同转站Pq＋1＝28。'],size=14,title_size=17,step=33)
    d.arrow([(410,1154),(410,1208)]);d.arrow([(1190,1154),(1190,1208)])
    d.box(50,1208,720,128,'当前标签：14时即可结算',
          ['真实转站14－快照中的启发式13＝＋1；归一化为0.1。',
           '先训练当前预测头和共享表示，无需等下一周期结束。'],size=13.5,title_size=17)
    d.box(830,1208,720,128,'下一周期标签：28时完成回填',
          ['真实就绪－24＝[－1，0，＋4，－2，0]，再除以10。',
           '未结算时保留快照、等待标签，不填零，也不提前训练。'],size=13.5,title_size=17)
    d.arrow([(410,1336),(410,1366),(30,1366),(30,665),(50,665)],dashed=True)
    d.arrow([(1190,1336),(1190,1366),(410,1366)],dashed=True)
    d.label(800,1366,'虚线：已结算标签训练在线预测头及共享编码／分层融合；固定副本不参与训练。',12.8)
    d.text(50,1402,'预测描述当前状态在后续执行中的时间结果；没有为每个候选动作分别构造反事实排程。',12.7)
    return d.save('工作三_分层融合与语义监督_具体示例_20261012')

if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True)
    a=hierarchy();b=example()
    assert max([8,13,10,7,9])==13
    assert max([8,14,10,7,9])==14
    assert [x-24 for x in [23,24,28,22,24]]==[-1,0,4,-2,0]
    print(json.dumps({'figures':[a,b],'numeric_consistency':'passed'},ensure_ascii=False))
