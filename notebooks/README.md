# notebooks

本目录只放探索性分析。约定如下:

- notebook **仅用于探索**,包代码永远不 import 它们。任何需要被复用的逻辑,都要落到 `src/shingan/` 并补上测试。
- 保持小而可重跑:每个 notebook 都能 top-to-bottom 一次跑完,不依赖隐藏状态、手工步骤或本地绝对路径。
- **提交时清空输出**。仓库通过 `pre-commit` 的 `nbstripout` 自动处理,不要手工粘贴图片或大段 stdout。
- 命名格式 `<two-digit-order>-<short-topic>.ipynb`,例如 `01-explore-synthetic.ipynb`。序号只表示阅读顺序,不代表依赖关系。
- 数据一律从 `data/processed/` 读取;本地没有数据时先跑 `shingan data synth`。

## 计划中的第一个 notebook

`01-explore-synthetic.ipynb`:用 `shingan data synth` 生成的数据,检查三类标签
(`default_risk` / `fraud_risk` / `tail_risk`)的分布、正样本比例与时间分布,确认标签口径和
as-of discipline 符合预期。若其中产生可复用的结论或检查逻辑,回填到 `src/shingan/` 并配测试。
