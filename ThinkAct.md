### 学術論文調査発表
# ThinkAct : Vision-Language-Action Reasoning via Reinforced Visual Latent Planning <br> 強化学習による視覚潜在プランニングを用いた Vision-Language-Action (VLA) 推論

Project page : https://jasper0314-huang.github.io/thinkact-vla/  
arXiv : https://arxiv.org/abs/2507.16815   
arXiv HTML page : https://arxiv.org/html/2507.16815v2  
code : coming soon(None) 

本研究発表は、台湾大学, Nvidia の共同発表で、2025年 7月 に発表されました。  

## これまでの VLA について

### 従来のVLAは **言語化された視覚** しか扱っていなかった

However, these models predict actions directly from vision and language inputs, often bypassing structured planning or intermediate reasoning.  
しかし、これらのモデルは、視覚と言語の入力から直接行動を予測するため、構造化された計画や中間的な推論をしばしば迂回します。

As a result, their capability to handle complex instructions, long-horizon goals, or out-of-distribution scenarios remains limited.  
その結果、複雑な指示、長期的な目標、または分布外のシナリオを処理する能力は限られたままです。

> 従来VLA (例：OpenVLA, TraceVLA, HAMSTER) は **"視覚をただの入力特徴として扱い、計画を内部的に持っていなかった"** 。つまり「見えている状況を理解して行動する」というより「言語→行動マッピング」止まりでした。


### CoT(Chain-of-thought)を導入しても、それは"言語的な思考の列"にとどまっていた

ECoT synthesizes intermediate subgoals via prompting and applies supervised fine-tuning to teach VLAs to reason before acting.   
ECoTはプロンプトで中間目標を作り、行動前に推論するようVLAを教師ありで訓練します。

However, they depend on either curated CoT supervision or task-specific video generation, limiting their scalability.   
しかし、それらはキュレーションされたCoT監督やタスク固有のビデオ生成に依存しており、スケーラビリティが制限されます。  

> CoT (Chain-of-Thought) 自体は **あくまでテキスト的推論** であり、AIが「視覚的に想像」するわけではなかった。

CoT について.  
https://www.promptingguide.ai/jp/techniques/cot ,   
https://note.com/npaka/n/n424192d89484  



## ThinkAct の提案
ThinkAct は、  
「マルチモーダルLLM (MLLM) が **視覚的潜在プラン (visual plan latent)** を生成し、それを下流の行動モデルに条件付けして実行する」ことで、 
**視覚に接地した長期計画 (long-horizon planning) と自己修正** を可能にする、  
強化学習で強化されたデュアルシステムVLAフレームワークです。


## ThinkActの開発的意義

開発者目線で見ると、ThinkActは以下のような設計的ブレイクスルーを提案しています：

| 従来のVLA | ThinkAct |
| -- | -- |
| 視覚を「言語に変換」して処理 | 視覚潜在空間を直接扱う (Visual Latent)  |
| 一枚画像ベースの単発行動 | 時系列的な潜在軌道を生成 |
| 完全教師ありSFT | 強化学習 (RL) で長期推論を学習 |
| end-to-endでblack-box | 推論と制御を分離し、連携 (Think + Act) |
| 現場での安定性が低い | few-shot適応、自己修正が可能 |

### アーキテクチャ(機械学習モデル構成) 
ThinkAct は 推論・制御の2系統 (dual-system) モデルであり、それぞれ独立したモデルとして学習・接続されています。 

#### 1. MLLM (Reasoner) 
役割
 - 入力：視覚 (画像 or 短い動画) とテキスト指示
 - 出力：潜在推論表現 (visual plan latent) 
 - 機能：環境を理解し、「どう動くか」を計画する
 - 学習：強化学習 (GRPO) ＋行動報酬 (visual reward) 

モデル詳細 (論文実装) 
| 項目 | 内容 |
| ------ | ---- |
| ベースモデル | **Qwen2.5-VL 7B** (マルチモーダルLLM)  |
| 出力形式 | 1) reasoning embedding $V_t$ <br> 2) visual plan latent $C_t$ |
| 特徴 | 画像＋テキストをエンコードし、自己回帰的に潜在表現を生成  |
| 強化学習 | Group Relative Policy Optimization (GRPO) |
| 報酬 | 行動整合報酬 (goal + trajectory) ＋出力形式の正しさ |
| 学習フェーズ | SFT (教師ありで初期化) → RLで推論改善 |


#### 2. Action Model (行動側：Act) 
役割
 - 入力：環境の視覚状態 + Thinkからの潜在計画 $C_t$
 - 出力：実行可能な行動 (例：グリッパー制御ベクトル) 
 - 機能：推論された計画に従って環境で正しく動く
 - 学習：模倣学習 (imitation learning) 

モデル詳細
| 項目 | 内容  |
| -- | --- |
| ベース  | **Diffusion Policy (DiT-based Policy)** |
| パラメータ数  | 約 4.32億 |
| 状態エンコーダ | DINOv2 (画像) ＋ CLIP (テキスト)  |
| 接続 | visual plan latent $C_t$ をQ-Former経由で入力に統合  |
| 学習データ | Open X-Embodiment (OXE) ＋ LIBERO |
| 学習方式 | imitation learning (教師あり、模倣デモを使用) |


## ThinkAct の何が新しいか？

### 1. Visual Latent Planning (視覚的潜在プランニング) 
従来の CoT (Chain-of-Thought) は「テキスト上の思考」でした。  
ThinkAct はこれを 「視覚潜在空間上の思考」 に変えた。  

技術的には：
 - 画像や動画から抽出された特徴を latent space (DINOv2やCLIPの埋め込み空間など) に写像。
 - この潜在空間上で "どんな動き (軌跡) をたどるべきか" をシーケンスとして生成。
 - 生成された latent plan を後段の Action Model に渡して制御。

これまでの技術用途：
 - 世界モデル (World Model, Dreamer, PlaNet など) で latent dynamics (潜在空間内の未来予測) として使用。
  > ただしそれはシミュレーションであり、「言語推論」と組み合わされたことはなかった。
 - ThinkAct はこれを VLA (Vision-Language-Action) ＋RL の中に初めて組み込んだ。


### 2. Action-Aligned Visual Rewards (行動整合視覚報酬) 
何が新しいか：  
LLMの「出力 (推論) 」を、視覚的・行動的に評価する報酬で訓練したことです。

従来のRLHF (例：ChatGPTの強化学習) では報酬は「人間の好み」や「テキスト正解度」でした。
ThinkAct はこれを視覚世界に持ち込み、次のようにしました：

 - $r_{goal}$ ： 開始・終了位置の一致度 (目標に近いか) 
 - $r_{traj}$ ： 軌跡分布の一致度 (動きの滑らかさ、一貫性) 

> つまり、「目で見て正しい動きか」を報酬化したという点が新しい。

これまでの技術用途：  
ロボット学習 (模倣・模倣＋RL) で reward shaping に使われてきた。
例：Distance-based reward, DTW (Dynamic Time Warping) で軌跡距離を測る。

ただし、それはポリシー (行動モデル) に対する報酬。

ThinkActでは **LLM側 (推論部分)** に報酬を与えて“考え方”を強化する。  
ここが革新的。

### 3. Dual-system / Asynchronous Operation (思考と行動の分離設計) 

何が新しいか：
「考えるAI」と「動くAI」を別々のモデルで実装し、非同期に動かす設計を取った点です。

 - Think (MLLM) は「今後のNステップを見通した推論」を行う。
 - Act (行動モデル) は「そのNステップを高速に実行」する。

これにより：

 - MLLMの推論を頻繁に呼ばずに済む (コスト削減) 。
 - 長期推論を視覚潜在表現として蓄え、効率的に実行可能。
 - 再プランニングも容易。

これまでの技術用途：

 - クラシカルロボティクスの「Planner (計画器) ＋Controller (制御器) 」の分離構造。 <br>例：RRTやAで経路計画し、PD制御器で追従。
 - 強化学習でも「high-level planner」＋「low-level policy」の階層型RL (HRL) に似ている。
 - ThinkAct はこの概念を LLM＋Actionモデル に適用した点が新しい。

#### 4. Reinforced Fine-Tuning（GRPO：Group Relative Policy Optimization）

何が新しいか：
ThinkActは GRPO（Group Relative Policy Optimization）を使い、
LLM出力の「良い推論」を視覚報酬で相対的に強化します。

従来のRLHF（ChatGPTなど）は：

 - 人間ラベルや報酬モデルを用いて、テキスト品質を強化する。

ThinkActは：

 - 実世界タスクにおける「目標達成度・軌跡一致度」を報酬にして、
 - それを LLMの潜在推論 に対して最適化。

> 言語モデルを「視覚的に行動する方向」へチューニングした、初のRL応用例。

これまでの技術用途：

 - 自然言語モデルのRLHF（OpenAI, Anthropicなど）で使われていた。
 - 強化学習では PPO（Proximal Policy Optimization）の派生系。
 - ThinkActではLLMの出力空間を行動的に評価して最適化に使ったのが革新。

#### ThinkAct が新しい理由と背景技術
| 新しい要素 | これまでの主用途 | ThinkActの革新点  |
| ----- | ---- | ------ |
| Visual Latent Planning | 世界モデル・潜在動力学（Dreamer, PlaNet） | 言語LLMの思考プロセスを視覚潜在空間で行う  |
| Action-Aligned Visual Rewards | ロボットRLの報酬設計（距離/軌跡一致）  | LLM推論の訓練に視覚報酬を導入 |
| Dual-System / Asynchronous  | 階層型RL、クラシカル制御構造  | Think（LLM）とAct（policy）を分離・非同期化 |
| GRPO (Reinforced Fine-tuning) | NLPのRLHF | 視覚タスクのLLM最適化に転用 |




----

## 実現された効果

ThinkActは、以下のような高度な動作能力を実現しています：

 - few-shot適応：数回のデモンストレーションで新しいタスクに適応可能。
 - 長期的な計画能力：複数ステップにわたる行動計画の生成と実行。
 - 自己修正機能：失敗を検出し、計画を修正してタスクを完了。
 - 動的環境への適応性：環境の変化に対応した柔軟な行動。

<video src="https://jasper0314-huang.github.io/thinkact-vla/static/videos/comparison_1.mp4" controls="true"></video>


<video src="https://jasper0314-huang.github.io/thinkact-vla/static/videos/comparison_3.mp4" controls="true"></video>

### few-shot適応(few-shot)

具体例：
 - 「コップを棚に置く」「スプーンを引き出しに入れる」といったタスクを2〜3回示すだけで、新しいタスク（例：フォークを同じ棚に置く）に適応可能。

ポイント：
 - デモは言語＋視覚（動画や画像）情報として提示される
 - MLLMがパターンを抽象化して、未学習タスクにも対応できる

Few-Shotプロンプティング について.  
https://www.promptingguide.ai/jp/techniques/fewshot

### 長期的な計画能力(multi-step planning)

「複数ステップ」とは、単純な一回の動作ではなく 5〜10ステップ以上の連続行動 を意味する。

具体例（RoboVQAベンチマーク）：
1. 冷蔵庫から牛乳を取り出す
2. カップを用意する
3. 牛乳をカップに注ぐ
4. こぼれないようにテーブルまで運ぶ
5. カップをテーブルに置く

特徴：
 - 中間ステップでの条件分岐（ドアが閉まっている → 開ける）が含まれる
 - DiT(Decision Transformer)ベースの行動モデルが「次の行動」を逐次決定することで実現


### 環境に柔軟に対応できる（dynamic environment adaptation）

事前に完全に固定された環境でなくても正しい行動を取れる 

具体例：
 - 目標物が予期せぬ位置にある場合でも、探索行動を追加して到達
 - 手元の障害物を避けて移動
 - 途中で物を落とした場合に計画を修正して再挑戦

証明：
 - LIBEROベンチマークで、物体の位置をランダム化した環境下でも成功率が高い
 - EgoPlan-Bench2では、人間が動かすオブジェクトに対しても計画を修正してタスク完了
