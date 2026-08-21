# 项目模型结果与比较

## Historical Average 总体结果
项目在 PEMS04 测试集上得到 Historical Average 的 MAE 26.4805、RMSE 43.5284、MAPE 16.5861%。这些数值来自 artifacts/historical_average_metrics.json。

## GRU 总体结果
项目当前最佳 GRU 在同一 PEMS04 测试集上得到 MAE 27.1363、RMSE 41.0972、MAPE 22.9937%。这些数值来自 artifacts/gru_best_metrics.json。

## MAE 对比
在当前实测中，Historical Average 的 MAE 26.4805 低于 GRU 的 27.1363，因此按平均绝对误差判断，Historical Average 略优。不能据此笼统宣称 GRU 全面优于历史平均。

## RMSE 对比
在当前实测中，GRU 的 RMSE 41.0972 低于 Historical Average 的 43.5284。这说明 GRU 在平方误差关注的大偏差方面表现更好，但仍需要结合 MAE 和 MAPE。

## MAPE 对比
在当前实测中，Historical Average 的 MAPE 16.5861% 低于 GRU 的 22.9937%。由于低流量位置会影响百分比误差，解释时还应结合项目的零值屏蔽规则。

## GRU 预测距离变化
GRU 的分预测步 MAE 从第 1 步约 18.7511 增加到第 12 步约 35.7175，显示预测距离越远，误差总体越大。该趋势来自 gru_best_metrics.json。

## Historical Average 预测距离变化
Historical Average 的 12 个分预测步 MAE 都约为 26.48，变化很小。这是因为模型主要依据时间桶平均值，而不是递推更新隐藏状态。

## 模型选择结论
当前结果没有单一模型在三个总体指标上全部占优。Historical Average 的 MAE 和 MAPE 更低，GRU 的 RMSE 更低；实际选择需要说明更重视平均误差、相对误差还是大误差。

## 结果适用范围
这些成绩只适用于当前 PEMS04 文件、项目数据处理和测试划分。它们不能直接作为昆明道路的预测精度，也不能替代其他论文在不同数据版本上的结果。

## 下一步模型扩展
STGCN、DCRNN、Graph WaveNet、STDN 和 HyperD 可以逐步加入统一评估，但只有在相同数据契约下实际复现并生成结果 JSON 后，才能写入对比表或简历。
