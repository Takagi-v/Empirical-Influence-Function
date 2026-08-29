# FIM 场景下 Spurious Correlation 与程序关系监督方法的完整实验设计

## 1. 研究动机与目标

### 1.1 核心问题

训练代码模型时，训练数据中会同时存在两类能够预测 target middle 的 pattern：

1. **Generalizable pattern（可泛化模式）**：由程序语义和结构产生，在训练环境与测试环境之间保持稳定。例如 def-use、reaching definition、data/control dependency、类型约束、作用域绑定以及 suffix 中对 middle 定义变量的后续使用。
2. **Spurious pattern（伪模式）**：在训练数据中恰好与 target 相关，却不是正确程序行为的决定因素。当数据来源、项目、格式或人为构造规则发生变化时，这种相关性可能消失甚至反转。例如注释标记、格式、identifier style、repository provenance、hole 位置或 serialization artifact。

我们真正希望模型学习的是第一类 pattern。模型即使在 IID 测试集上表现很好，也可能只是利用第二类 pattern 预测了正确 token；这种模型在训练分布改变后将无法可靠泛化。

因此，本实验的核心目的不是简单验证“某个注释是否会影响模型”，而是构造一个同时包含两类竞争证据的可控 FIM 环境：

```text
Generalizable program pattern G ────────> 正确 target Y

Spurious pattern Z ── 训练集中相关 ─────> 正确 target Y
                   但不改变程序语义
```

在训练集中，$G$ 和 $Z$ 通常都指向同一个 target，因此仅观察训练或 IID 测试性能无法判断模型实际依赖了哪一种 pattern。在反事实测试中保持 $G$ 和正确 target 不变，只翻转或删除 $Z$，才能识别模型是否学习了 spurious correlation。

### 1.2 Cue 与 spurious correlation 的关系

本文中的 **cue 是 spurious pattern 的一个可观察、可干预载体，但 cue 本身不等于 spurious correlation**。

例如：

```python
# build-shard: cedar
```

这一行注释只是 cue。真正被人为注入的 spurious correlation 是：在训练集中，`cedar` 高概率与 `value_a` target 同时出现，而 `maple` 高概率与 `value_b` target 同时出现。

它之所以是 spurious，必须同时满足：

1. cue 与 target 在训练分布中具有较强统计相关性；
2. cue 不参与程序的 def-use、data/control flow、类型或执行语义；
3. 将 `cedar` 替换成 `maple` 时，正确 middle 和完整程序行为都不应改变；
4. 这种 cue-target 映射不是部署环境中稳定成立的规则。

因此，应严格区分：

- **Cue $Z$**：注释、格式、名字风格等输入特征；
- **Spurious correlation**：训练分布中 $Z$ 与 target $Y$ 之间存在、但不具有程序语义稳定性的相关关系；
- **Shortcut learning**：模型在预测时实际利用了这种关系；
- **Robustness failure**：在保持程序语义不变、只干预 $Z$ 后，模型预测发生错误变化。

人工 cue 的价值是使 spurious correlation 的强度、方向和位置都可以精确控制。后续可以把人工 cue 替换为更自然的 identifier style、formatting 或 provenance pattern，检验结论是否具有外部有效性。

### 1.3 形式化目标

给定完整程序：

$$
C_i=P_i\Vert M_i\Vert S_i,
$$

其中 $P_i$、$M_i$、$S_i$ 分别为 prefix、middle 和 suffix。令：

- $Y_i$：正确 middle 或其中的关键 target token；
- $G_i$：由程序分析确认、与 $Y_i$ 存在稳定程序关系的 context pattern；
- $Z_i$：不改变程序语义的 spurious cue。

受控数据应满足：

$$
Y_i=f(G_i),
$$

并且对任意 cue 取值：

$$
Y_i\big(do(Z_i=0)\big)=Y_i\big(do(Z_i=1)\big).
$$

但在训练数据采样过程中人为设置：

$$
P_{\mathrm{train}}(Z_i=Y_i)=\rho,\quad \rho>0.5.
$$

也就是说，$G$ 是稳定且与正确程序行为相关的 pattern，$Z$ 只是训练环境中容易利用的预测信号。测试时通过改变 $P(Z\mid Y)$，同时保持 $P(Y\mid G)$ 不变，测量模型到底依赖哪一种 pattern。

模型输入和监督形式统一为：

```text
<fim_prefix> P <fim_suffix> S <fim_middle> M <eot>
------------------------------------------ ^^^^^^^
          label = -100                    有监督
```

训练目标是仅在 middle 和 EOT 上计算 next-token prediction loss。本方案不将 NTP 与 SFT 作为两种训练方式比较。

### 1.4 本实验需要回答的问题

1. 当 $G$ 与 $Z$ 在训练集中同时预测正确 target 时，普通 FIM 模型在多大程度上会选择更容易利用的 $Z$，而不是稳定的程序 pattern $G$？
2. 当测试环境反转或删除 $Z$ 时，这种 shortcut reliance 会造成多大的性能下降？
3. 程序分析能否为 target token 标注出代表 $G$ 的高置信 context tokens？
4. 在 next-token prediction loss 之外加入关系 contrastive loss，能否使模型的 attention/saliency 从 $Z$ 和无关 token 转移到 $G$？
5. 这种内部依赖变化能否真正转化为 conflicting、neutral 和 worst-group FIM 性能的提高，而不仅是 attention 数值更符合标注？

## 2. 核心假设

### H1：训练分布中的 spurious opportunity 会导致 shortcut learning

当训练集满足

$$
P_{\mathrm{train}}(Z=Y)=\rho,\quad \rho\gg 0.5,
$$

即使 $G$ 已经足以确定正确 target，普通 FIM 模型仍可能因为 $Z$ 更短、更局部或更容易优化，而学习 $Z\rightarrow Y$ 的 shortcut。

预期表现为：aligned test 性能较高，而保持 $G,Y$ 不变、令 $Z$ 与 $Y$ 冲突后，性能明显下降。

### H2：程序关系监督提高对 generalizable pattern 的依赖

程序分析标注出的 relation-positive token 是 $G$ 的 token-level operationalization。加入关系 contrastive loss 后，模型在预测 target token 时对这些 token 的 attention/saliency 应提高，对 hard distractor 和 spurious cue 的依赖应下降。

这里的方法目标不是学习某一种 cue 的黑名单，而是学习一种更一般的归纳偏置：优先利用由程序结构和语义支持的关系。

### H3：内部 relation alignment 应转化为分布外稳健性

内部 relation alignment 的提高应转化为行为层面的选择性：

- 在不改变 $G$ 和程序语义、只翻转 $Z$ 时，模型预测应保持稳定；
- 在真正改变或破坏 $G$ 时，模型应表现出更强敏感性；
- 对 identifier 做语义一致的 alpha-renaming 时，模型输出应产生正确的等变变化。

因此，有效性必须同时由 internal alignment、causal intervention 和 conflicting/worst-group performance 三类证据支持。

## 3. 推荐的第一版任务：变量绑定与 def-use FIM

第一版不建议对任意随机 middle 直接加入 cue。更干净的构造是从 CodeSearchNet Python 函数中抽取具有明确 def-use 关系的局部变量补全任务。

### 3.1 Target 候选

选择一个局部变量的定义位置作为 target，例如：

```python
def process(items):
    value_a = normalize(items)
    <HOLE> = transform(value_a)
    return value_b
```

完整程序中的 middle 是：

```python
value_b
```

程序分析可建立：

```text
middle 中 value_b 的定义  ->  suffix 中 return value_b 的使用
```

若错误补为 `value_a`，则会覆盖另一变量，并使 suffix 中的 `value_b` 缺少正确 reaching definition。因此 suffix 中 `value_b` 是预测 target 的高置信程序关系 token。

### 3.2 候选过滤规则

候选 target 最好满足：

- target 是简单局部变量定义，例如 `Name(Store)`，而不是 attribute、global 或复杂 destructuring；
- target 是该变量在当前路径上的第一个 reaching definition；
- suffix 中至少存在一次由该定义到达的 use；
- target 与 suffix use 之间不存在可能重新定义同一变量的路径；
- 函数中存在另一个无直接 def-use/data-dependence 的局部变量作为 hard distractor；
- target 与 distractor 最好具有相同或兼容的静态类型、相同 AST 类别和相似出现频率；
- 排除 `eval`、`exec`、`locals()`、`globals()`、反射式 `getattr/setattr`、字符串化变量名等会破坏 alpha-renaming 等价性的代码；
- 第一版排除闭包、复杂 comprehension scope、动态 import 和无法可靠解析的代码。

### 3.3 Canonical alpha-renaming

将 target binding 和 distractor binding 以符号绑定为单位进行 alpha-renaming，而不是字符串替换：

- 一个绑定统一重命名为 `value_a`；
- 另一个绑定统一重命名为 `value_b`；
- 随机决定哪个名字属于正确 target binding。

定义：

$$
Y=0\Longleftrightarrow M=\texttt{value\_a},\qquad
Y=1\Longleftrightarrow M=\texttt{value\_b}.
$$

这样每个样本中 `value_a` 和 `value_b` 都存在，模型不能仅根据某个名字是否出现来完成任务，而必须识别哪个 suffix use 与 hole 中的 definition 构成正确的程序关系。

重命名应使用 CST/AST symbol binding 信息完成，随后检查：

- 重命名前后 AST 除 identifier 外等价；
- 完整程序仍可解析；
- target def-use edge 仍存在；
- distractor 不在 target 的高置信程序依赖闭包中。

## 4. Spurious correlation 注入

### 4.1 Cue 定义

第一版使用不影响 Python 运行语义的注释 cue：

```python
# build-shard: cedar
# build-shard: maple
```

定义：

$$
Z=0\Longleftrightarrow\texttt{cedar},\qquad
Z=1\Longleftrightarrow\texttt{maple}.
$$

训练集按以下方式采样：

$$
Z=
\begin{cases}
Y,&\text{概率 }\rho,\\
1-Y,&\text{概率 }1-\rho.
\end{cases}
$$

建议至少包含：

- $\rho=0.5$：cue 与 target 独立；
- $\rho=0.8$：中等相关；
- $\rho=0.95$：强相关；
- 可选 $\rho=1.0$：完美相关。

第一轮可只跑 $0.5$ 与 $0.95$，确认现象后再画完整 dose-response curve。

### 4.2 注入约束

- cue 只能位于 prefix 或 suffix，不能进入 middle；
- 两种 cue 在目标 tokenizer 下的 token 数应一致；
- cue 的缩进、位置和距 hole 的距离必须一致；
- `cedar`、`maple` 在 $Y=0/1$ 中边际频率应相同；
- 在不同实验重复中反转 cue-to-label mapping，以消除 base model 对词本身的先验；
- 程序分析图中 cue 不应与 target 存在边；
- 关系 loss 的 negative sampler 不得根据“这是人工 cue”进行特殊处理，防止方法获得额外的人工标记信息。

### 4.3 FIM-specific 位置条件

第一主实验固定 cue 在 prefix 中的合法 statement boundary。随后做位置消融：

- cue 在 prefix；
- cue 在 suffix；
- cue 距 hole 近；
- cue 距 hole 远。

同时将真正 relation-positive token 分为 prefix-only、suffix-only 和 both-side，检验 PSM 序列化下可能存在的 suffix proximity bias。

## 5. 数据划分与反事实测试集

### 5.1 划分顺序

1. 按 repository 划分 train/dev/test；
2. 完成去重或近重复检查；
3. 在每个 split 内抽取 candidate；
4. 对 candidate 做 canonical renaming；
5. 生成 cue variant；
6. 生成 FIM 序列和 token-level relation labels。

同一 `base_id` 的所有重命名、cue 和 hole variant 必须属于同一个 split。测试集中可以包含同一 base 的多个反事实 variant，但这些 variant 不能出现在训练集中。

### 5.2 测试条件

对每个 test base 至少生成三个版本：

| 条件 | Cue | 程序语义 |
|---|---|---|
| Aligned | $Z=Y$ | 不变 |
| Conflicting | $Z=1-Y$ | 不变 |
| Neutral | 删除 cue | 不变 |

额外诊断条件：

- Unseen cue：使用训练中未见但等长的新 tag；
- Cue relocation：把同一 cue 从 prefix 移到 suffix 或反之；
- Distractor rename：只重命名无关 binding，正确 target 应保持等变或不变；
- Relation intervention：删除、遮蔽或一致重命名真正 relation-positive token，模型应比面对 cue flip 时更敏感。

### 5.3 建议规模

Pilot：

- Train：4,000–8,000 个 base examples；
- IID dev：500–1,000；
- Test：1,000 个 base examples；
- 每个 test base 生成 aligned、conflicting、neutral 三个版本；
- 至少 3 个随机种子。

若高置信 def-use candidate 不足，宁可降低规模，也不要用大量噪声 relation label。

## 6. 程序分析关系标注

### 6.1 标注时机

应先在完整程序 $P\Vert M\Vert S$ 上运行程序分析，得到 program dependence information，再将 middle 隐去并映射为 FIM token positions。

对 middle 中可可靠标注的 target token $y_j$，构造：

- $\mathcal P_j$：relation-positive context tokens；
- $\mathcal N_j$：高置信 negative candidates；
- $r_{j,k}$：relation type，如 def-use、data、control 或 type dependency。

关系监督只作用于具有高置信关系标注的 anchor target token，例如 identifier、operator、API callee、attribute 或 literal。普通语法 token、标点和 EOT 仍接受 next-token prediction loss，但不强制使用 relation loss。

### 6.2 第一版 relation types

优先使用精度最高的关系：

1. Local def-use / reaching-definition；
2. Middle definition 到 suffix use；
3. Prefix definition 到 middle use；
4. 同一 binding 的符号一致性；
5. 简单 RHS data dependency。

后续再加入：

- control dependency；
- receiver type 与 attribute/API resolution；
- import/call dependency；
- 跨函数或跨文件关系。

### 6.3 Source span 到 subword token

程序分析关系通常位于 AST/CST source spans，而模型使用 subword token。数据处理必须保存：

- source character/byte span；
- clean-code token span；
- FIM serialization 后的 token indices；
- identifier 被拆成多个 subword 时的聚合规则。

对于多 subword identifier，可以将所有 subword 视为一个 group，对 attention/saliency 求和或平均，避免长 identifier 因 token 数较多而获得不公平优势。

### 6.4 Negative sampling

“未被程序分析标注”不等于真正无关，尤其是注释、自然语言语义和分析器无法处理的动态依赖。因此不建议把所有 unannotated token 都作为同等强度的负样本。

推荐 negative pool：

- 与 target 同作用域但无 PDG 路径的 distractor binding；
- 与 positive token AST 类别和 token type 相同的 token；
- 与 positive token 距 target 距离匹配的 token；
- 程序分析确认不在 $K$-hop dependence closure 中的 token。

cue 可以自然进入这一 negative pool，但不能被强制每次选中。否则实验只能证明“显式告诉模型不要看 cue”有效，而不能证明通用程序关系监督有效。

## 7. 模型与损失函数

### 7.1 基础 next-token prediction loss

设 $J_i$ 为 middle 和 EOT 的 token positions：

$$
\mathcal L_{\mathrm{NTP}}
= -\frac{1}{|J_i|}\sum_{j\in J_i}
\log p_\theta\!\left(y_j \mid P_i, S_i, y_{<j}\right).
$$

prefix、suffix 和 `<fim_middle>` 之前的 label 全部设为 `-100`。

### 7.2 Relation score

对预测 target token $y_j$ 的 query position 与 context token $k$，定义 relation score：

$$
\phi_{j,k}
= \operatorname{Aggregate}_{l,h}\!\left(
\operatorname{AttentionOrSaliency}^{\,l,h}_{j,k}
\right).
$$

可以使用：

- 指定层/头的 raw attention；
- attention rollout；
- attention × gradient；
- target logit 对 context representation 的 gradient saliency；
- 已有方法中定义的其他可微 saliency。

若直接优化 gradient-based saliency，需要明确是否产生二阶梯度及其计算开销。主实验中应固定一种实现，其他形式作为消融，避免把 relation objective 与 saliency estimator 的差异混在一起。

### 7.3 Multi-positive contrastive loss

一种可实现的多正样本 InfoNCE 形式为：

$$
\mathcal L_{\mathrm{rel}}
= -\frac{1}{|A_i|}\sum_{j\in A_i}
\log\!\left(
\frac{
\sum_{p\in\mathcal P_j}\exp\!\left(\phi_{j,p}/\tau\right)
}{
\sum_{p\in\mathcal P_j}\exp\!\left(\phi_{j,p}/\tau\right)
+ \sum_{n\in\mathcal N_j}\exp\!\left(\phi_{j,n}/\tau\right)
}
\right).
$$

其中 $A_i$ 是具有可靠 relation labels 的 anchor target tokens，$\tau$ 是 temperature。

若希望每个 positive 都高于 hard negatives，可增加 pairwise margin ranking：

$$
\mathcal L_{\mathrm{rank}}
= \frac{1}{|A_i|}\sum_{j\in A_i}
\frac{1}{|\mathcal P_j|\,|\mathcal N_j|}
\sum_{p\in\mathcal P_j}\sum_{n\in\mathcal N_j}
\left[m-\phi_{j,p}+\phi_{j,n}\right]_+.
$$

建议第一版只选择一种 relation loss；两者并用应作为后续消融。

### 7.4 总损失

$$
\mathcal L
= \mathcal L_{\mathrm{NTP}} + \lambda\mathcal L_{\mathrm{rel}}.
$$

如有多种 relation type，可进一步设置：

$$
\mathcal L_{\mathrm{rel}}
= \sum_r w_r\,\mathcal L_{\mathrm{rel}}^{(r)}.
$$

第一版推荐只使用 def-use 关系并设置单一 $\lambda$，避免关系类型权重带来额外自由度。$\lambda$ 和 $\tau$ 应在独立的 $\rho=0.5$ development split 上确定，随后冻结；不要使用 conflicting test 或其近似数据调参。

## 8. 实验组

### 8.1 必要训练组

| 编号 | 数据 | 训练目标 | 作用 |
|---|---|---|---|
| A | No-cue | $\mathcal L_{NTP}$ | 干净基准 |
| B | Random-cue，$\rho=0.5$ | $\mathcal L_{NTP}$ | 控制加入额外注释本身的影响 |
| C | Biased-cue，$\rho=0.95$ | $\mathcal L_{NTP}$ | 标准 shortcut baseline |
| D | Biased-cue，$\rho=0.95$ | $\mathcal L_{NTP}+\lambda\mathcal L_{rel}$ | 提出方法 |
| E | Biased-cue，$\rho=0.95$ | shuffled relation labels | 排除额外正则与计算量带来的收益 |
| F | Counterfactual-balanced，$\rho=0.5$ | $\mathcal L_{NTP}$ | 数据去偏上界/参考线 |

其中 E 应保持与 D 相同的 relation 数量、正负样本数量、loss scale 和计算流程，只在 batch 内或同 relation type 内打乱 positive relation labels。

### 8.2 公平性控制

- 所有模型使用同一初始化 checkpoint；
- 使用同一 FIM tokenizer 和 special token；
- 使用相同 base examples、batch order、训练步数和 middle-token budget；
- checkpoint 按固定 step 或 IID dev NTP loss 选择；
- 至少 3 个 seed，推荐 5 个；
- 报告额外显存、训练时间和 backward 开销；
- 若 proposed loss 需要额外 forward/backward，应加入 compute-matched regularization baseline 或至少报告等 token budget 与等 wall-clock 两种比较。

## 9. 评测指标

### 9.1 主要行为指标

1. **Aligned target accuracy / Pass@1**；
2. **Conflicting target accuracy / Pass@1**；
3. **Neutral target accuracy / Pass@1**；
4. **Aligned–Conflict Gap**：

$$
\mathrm{SCGap}
= \operatorname{Acc}_{\mathrm{aligned}}
- \operatorname{Acc}_{\mathrm{conflicting}}.
$$

5. **Worst-group accuracy**：

$$
\operatorname{WorstGroupAcc}
= \min_{y,z}\operatorname{Acc}(Y=y, Z=z).
$$

6. **Paired Cue Flip Rate**：同一 base 只翻转 cue 后预测发生变化的比例；
7. **Functional correctness**：若存在可执行测试，报告 unit-test Pass@1，而不是只报告 exact match。

### 9.2 Forced-choice likelihood

对每个样本保存 gold middle $M_i$ 和由 distractor 构成的合法反事实 middle $M_i^-$：

$$
s_i(z)
= \frac{\log p\!\left(M_i \mid x_i^z\right)}{|M_i|}
- \frac{\log p\!\left(M_i^- \mid x_i^z\right)}{|M_i^-|}.
$$

定义 cue causal effect：

$$
\operatorname{CueEffect}
= \mathbb E_i\!\left[
s_i\!\left(z_{\mathrm{aligned}}\right)
- s_i\!\left(z_{\mathrm{conflicting}}\right)
\right].
$$

提出方法应显著降低 CueEffect，同时保持或提高 neutral 条件下的 gold preference。

### 9.3 内部机制指标

- Positive relation saliency mass；
- Hard-negative saliency mass；
- Cue saliency mass；
- positive-vs-negative ranking AUC；
- relation Precision@K / Recall@K；
- 不同 relation distance、prefix/suffix side、layer/head 上的分布。

不能只以 attention alignment 作为方法有效的证据。因为 attention 可以被训练成符合标签，但不一定真正影响输出。

### 9.4 Causal faithfulness 指标

对同一 example 做三类 intervention：

1. 删除或 mask relation-positive token；
2. 删除或 mask 距离、类型匹配的 negative token；
3. 翻转或删除 spurious cue。

记录 gold log-probability 的下降：

$$
\Delta_{\mathrm{pos}},\qquad
\Delta_{\mathrm{neg}},\qquad
\Delta_{\mathrm{cue}}.
$$

理想结果为：

$$
\Delta_{\mathrm{pos}} > \Delta_{\mathrm{neg}},
\qquad
\Delta_{\mathrm{pos}} > \Delta_{\mathrm{cue}}.
$$

更严格的测试是对 positive identifier 及其 target 做一致 alpha-renaming，模型应产生相应重命名后的 target，即表现为 equivariance，而不是对 token 字符串的机械依赖。

## 10. 统计分析

- 所有 aligned/conflicting 比较均以同一 `base_id` 做 paired analysis；
- confidence interval 使用按 repository 或 base program cluster 的 bootstrap；
- target-choice accuracy 可使用 paired bootstrap 或 McNemar test；
- 多 seed 报告 seed mean、standard deviation 和 pooled paired effect；
- 多 relation type、距离分桶和多指标检验应进行多重比较校正；
- 主指标和成功阈值应在实验前固定。

建议的成功标准：

1. D 相对 C 显著提高 conflicting accuracy 和 worst-group accuracy；
2. D 显著减小 SCGap 和 Cue Flip Rate；
3. D 的 neutral accuracy 不出现明显退化；
4. D 提高 positive relation AUC/mass，并降低 cue mass；
5. D 的 $\Delta_{pos}-\Delta_{neg}$ 显著增大；
6. shuffled-label 组 E 不能复现 D 的主要收益。

可以预注册例如“conflicting accuracy 至少提高 5 个百分点或相对 error 降低 20%，neutral accuracy 下降不超过 1–2 个百分点”，但最终阈值应结合模型和 pilot 难度确定。

## 11. 关键消融实验

按优先级建议：

1. $\lambda=0$ 与不同 $\lambda$；
2. 正确 relation vs shuffled relation；
3. def-use only vs lexical same-identifier supervision；
4. hard matched negatives vs random negatives；
5. raw attention vs 当前 saliency estimator；
6. cue 位于 prefix vs suffix；
7. relation evidence 位于 prefix vs suffix；
8. relation distance bins；
9. $\rho\in\{0.5,0.8,0.95,1.0\}$；
10. identifier hole、expression hole、statement hole；
11. 只监督某一层/头 vs 多层多头聚合；
12. 程序分析高置信子集 vs 扩大但更噪声的 relation set。

最关键的三个消融是 shuffled labels、hard-negative matching 和 cue/causal-token intervention。它们分别排除普通正则化收益、位置/词类捷径，以及“attention 看起来正确但实际不影响预测”的问题。

## 12. 数据记录格式

每条数据建议至少保存：

```json
{
  "base_id": "repo/path/function/span",
  "repo_id": "owner/repo",
  "split": "train",
  "prefix": "...",
  "middle": "value_a",
  "suffix": "...",
  "negative_middle": "value_b",
  "target_class": 0,
  "spur_class": 0,
  "rho": 0.95,
  "alignment": "aligned",
  "cue_span": [120, 125],
  "anchor_target_spans": [[210, 217]],
  "positive_context_spans": [[260, 267]],
  "negative_candidate_spans": [[85, 92]],
  "relation_types": ["def_use"],
  "analysis_confidence": "high",
  "fim_serialization": "..."
}
```

建议 source spans 与 tokenizer indices 分开保存，使同一分析结果可以映射到不同 tokenizer，而不必重新运行程序分析。

## 13. 数据质量检查

生成 pipeline 应自动检查：

- `prefix + middle + suffix` 能恢复完整 canonical program；
- cue 注入前后程序 AST 仅有 comment 差异；
- alpha-renaming 前后符号绑定关系保持；
- gold middle 插回后可解析；
- distractor middle 插回后仍可解析，以支持 forced-choice 公平比较；
- positive relation 至少有一个位于可见 prefix/suffix；
- cue 不在 positive relation closure 中；
- cue 不进入 middle；
- target/distractor 与两种 cue 的 tokenizer 长度匹配；
- 每个 target class、relation type、repo、长度桶中的 aligned/conflicting 比例符合设计；
- train/dev/test 无 base、repo 或 near-clone 泄漏；
- 对 test 和一定比例的 train examples 做人工审查。

## 14. 分阶段执行路线

### Phase 0：实现单元测试

用几十个人工小程序验证：

- FIM offset 映射；
- def-use 标注；
- alpha-renaming；
- cue 注入；
- relation loss 的正负方向；
- intervention 指标。

### Phase 1：受控 CodeSearchNet def-use benchmark

- 只使用 Python；
- 只使用 local variable definition-to-suffix-use；
- identifier-level hole；
- comment cue；
- $\rho=0.5$ 和 $0.95$；
- 完成 A–F 训练组和主要机制分析。

这一阶段用于回答：方法能否在最可靠的程序关系上抵抗可控 shortcut。

### Phase 2：扩展自然性

- expression/statement hole；
- data/control/type/API relations；
- identifier、format、position 等不同 cue；
- cue 和 causal evidence 的远近、prefix/suffix 位置变化；
- 加入可执行测试或 test-bearing repositories。

### Phase 3：泛化验证

- 在一种 cue 上训练、另一种 spurious feature 上测试；
- 跨 repository、代码风格和 target family；
- 检验 relation supervision 是否提高一般 FIM completion，而不仅针对固定 tag。

## 15. 预期结果与解释边界

最有说服力的结果组合是：

1. 普通模型在 biased training 后出现明显 aligned–conflict gap；
2. 提出方法提高 conflicting 和 worst-group performance；
3. neutral/clean performance 基本保持；
4. positive-token saliency 增加、cue saliency 降低；
5. 对 positive token 的 causal intervention 效果增强；
6. shuffled relation labels 无法产生同样收益。

如果只有 attention/saliency 指标改善而行为性能不变，结论只能是“模型内部 attribution 更符合程序分析标签”，不能声称模型真正减少了 shortcut reliance。

如果 conflicting performance 提高但 shuffled labels 也同样提高，更可能是额外正则化或优化效应，而不是程序关系标注本身有效。

如果 proposed model 在 cue flip 上稳定，但对 relation-positive token 的干预也不敏感，可能只是整体置信度下降或模型忽略了两类信号，需要结合 neutral performance、log-odds 和 causal intervention 判断。

最终应把有效性定义为三层证据同时成立：

```text
正确程序关系标注
        ↓
内部 relation alignment 提升
        ↓
对 causal token 更敏感、对 cue 更不敏感
        ↓
conflicting / worst-group FIM 正确率提升
```
